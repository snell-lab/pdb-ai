"""The density columns, with what each holds and when it is missing.

The value columns and the atom status are atom columns. The two
root-mean-square densities, the resolution, the sample rate, and the map
status say how the entry's maps were made, so they are structure columns.

A value column ending in _sigma is in units of its own map's root-mean-square
density, so two of them have no common scale. The columns ending in _eA3 hold
the difference density and the model's calculated density on the one scale
of the map coefficients, so that one over the other is a fraction. Only
PDB-Redo's file holds calculated structure factors, so only its family has
them.

The gradient columns hold the slope of the difference map at an atom along
the x, y, and z axes of the model's frame, in sigma per Å. They are read only
at the position of the model each family's maps were calculated about, since
the slope of a difference map says where the data would move that model's
atom. PDB-Redo's family also holds the shift of the atom that would level the
slope, in Å, which divides the slope by the curvature of the calculated map.

There are two families, one for each source of maps, and a run writes the
family of each source it reads. The columns of PDB-Redo's maps were named
first and carry no source in their names, except the two read at PDB-Redo's
own atoms. Every column of the wwPDB validation maps ends in _wwpdb.
"""

from __future__ import annotations

from ...core import (
    PROFILE_RADIATION_SENSITIVITY,
    ColumnGranularity,
    ColumnSpec,
    EngineOptions,
)
from ...sources.map_sources import (
    DENSITY_SOURCE_PDB_REDO,
    DENSITY_SOURCE_WWPDB,
    density_source_selected,
)
from .constants import (
    ATOM_STATUS_NO_COORDINATES,
    ATOM_STATUS_NO_PDB_REDO_ATOM,
    ATOM_STATUS_OK,
    MAP_FC_LABELS,
    MAP_MAX_GRID_POINTS,
    MAP_MIN_SAMPLE_RATE,
    MAP_SAMPLE_RATE,
    MAP_STATUS_CELL_MISMATCH,
    MAP_STATUS_NO_COEFFICIENTS,
    MAP_STATUS_NO_ENTRY,
    MAP_STATUS_NO_VALIDATION_MAPS,
    MAP_STATUS_NOT_PAIRED,
    MAP_STATUS_OK,
    MAP_STATUS_TOO_LARGE,
)

GROUP = "Density Fields"

# The values at the deposited position, the 2mFo−DFc map first.
DEPOSITED_VALUE_COLUMNS = ("density_2fofc_sigma", "density_fofc_sigma")
# The values at the position PDB-Redo's model gives the atom, in the same order.
PDB_REDO_VALUE_COLUMNS = (
    "density_2fofc_sigma_pdb_redo",
    "density_fofc_sigma_pdb_redo",
)
# The difference density and the calculated density on the scale of the map
# coefficients, at the deposited position and at PDB-Redo's.
DEPOSITED_ABSOLUTE_COLUMNS = ("density_fofc_eA3", "density_fc_eA3")
PDB_REDO_ABSOLUTE_COLUMNS = (
    "density_fofc_eA3_pdb_redo",
    "density_fc_eA3_pdb_redo",
)
# The gradient of the difference map at the position PDB-Redo's model gives the
# atom, along x, y, and z, and the shift of the atom that would level it.
PDB_REDO_GRADIENT_COLUMNS = (
    "density_fofc_gradient_x_sigma_per_A_pdb_redo",
    "density_fofc_gradient_y_sigma_per_A_pdb_redo",
    "density_fofc_gradient_z_sigma_per_A_pdb_redo",
)
PDB_REDO_SHIFT_COLUMNS = (
    "density_fofc_shift_x_A_pdb_redo",
    "density_fofc_shift_y_A_pdb_redo",
    "density_fofc_shift_z_A_pdb_redo",
)
ATOM_STATUS_COLUMN = "density_status"
# The root-mean-square density of each map, in the same order.
RMS_COLUMNS = ("density_2fofc_rms", "density_fofc_rms")
MAP_RESOLUTION_COLUMN = "density_map_resolution_A"
MAP_SAMPLE_RATE_COLUMN = "density_map_sample_rate"
MAP_STATUS_COLUMN = "density_map_status"

# The columns of the wwPDB validation maps, which are read at the deposited
# position only, since they were calculated about the deposited model.
WWPDB_SUFFIX = "_wwpdb"
WWPDB_VALUE_COLUMNS = tuple(f"{name}{WWPDB_SUFFIX}" for name in DEPOSITED_VALUE_COLUMNS)
# The gradient of the validation difference map at the deposited position.
WWPDB_GRADIENT_COLUMNS = (
    "density_fofc_gradient_x_sigma_per_A_wwpdb",
    "density_fofc_gradient_y_sigma_per_A_wwpdb",
    "density_fofc_gradient_z_sigma_per_A_wwpdb",
)
WWPDB_ATOM_STATUS_COLUMN = f"{ATOM_STATUS_COLUMN}{WWPDB_SUFFIX}"
WWPDB_RMS_COLUMNS = tuple(f"{name}{WWPDB_SUFFIX}" for name in RMS_COLUMNS)
WWPDB_MAP_RESOLUTION_COLUMN = f"{MAP_RESOLUTION_COLUMN}{WWPDB_SUFFIX}"
WWPDB_MAP_SAMPLE_RATE_COLUMN = f"{MAP_SAMPLE_RATE_COLUMN}{WWPDB_SUFFIX}"
WWPDB_MAP_STATUS_COLUMN = f"{MAP_STATUS_COLUMN}{WWPDB_SUFFIX}"

_PDB_REDO_STRUCTURE_COLUMNS = (
    *RMS_COLUMNS,
    MAP_RESOLUTION_COLUMN,
    MAP_SAMPLE_RATE_COLUMN,
    MAP_STATUS_COLUMN,
)
_WWPDB_STRUCTURE_COLUMNS = (
    *WWPDB_RMS_COLUMNS,
    WWPDB_MAP_RESOLUTION_COLUMN,
    WWPDB_MAP_SAMPLE_RATE_COLUMN,
    WWPDB_MAP_STATUS_COLUMN,
)
_STRUCTURE_COLUMNS = (*_PDB_REDO_STRUCTURE_COLUMNS, *_WWPDB_STRUCTURE_COLUMNS)
_TEXT_COLUMNS = (
    ATOM_STATUS_COLUMN,
    MAP_STATUS_COLUMN,
    WWPDB_ATOM_STATUS_COLUMN,
    WWPDB_MAP_STATUS_COLUMN,
)

# Every column of each family, in the order compute.py returns them.
PDB_REDO_COLUMN_NAMES = (
    *DEPOSITED_VALUE_COLUMNS,
    *PDB_REDO_VALUE_COLUMNS,
    *DEPOSITED_ABSOLUTE_COLUMNS,
    *PDB_REDO_ABSOLUTE_COLUMNS,
    *PDB_REDO_GRADIENT_COLUMNS,
    *PDB_REDO_SHIFT_COLUMNS,
    ATOM_STATUS_COLUMN,
    *_PDB_REDO_STRUCTURE_COLUMNS,
)
WWPDB_COLUMN_NAMES = (
    *WWPDB_VALUE_COLUMNS,
    *WWPDB_GRADIENT_COLUMNS,
    WWPDB_ATOM_STATUS_COLUMN,
    *_WWPDB_STRUCTURE_COLUMNS,
)

_SIGMA = (
    "in units of the map's root-mean-square density over the unit cell, with "
    "the map's mean subtracted"
)
_MISSING = "Missing when the entry has no maps, and density_status then says why."
_NO_CALCULATED = (
    "Also missing when the reflection file holds no usable calculated "
    f"structure factors ({MAP_FC_LABELS[0]}, {MAP_FC_LABELS[1]}), which the "
    "run report's density_message then says"
)

DESCRIPTIONS: dict[str, str] = {
    "density_2fofc_sigma": (
        "Density of PDB-Redo's 2mFo−DFc map at the atom's deposited position, "
        f"{_SIGMA}. It says how much density the data and PDB-Redo's model "
        "place where the depositor put the atom. Read by cubic interpolation, "
        "and not thresholded. The value depends on resolution, since a "
        "sharper map has higher peaks, so compare it within bins of "
        f"resolution. {_MISSING} Also missing when the deposited position is "
        "not a finite point."
    ),
    "density_fofc_sigma": (
        "Density of PDB-Redo's mFo−DFc difference map at the atom's deposited "
        f"position, {_SIGMA}. A negative value says PDB-Redo's model places "
        "more density there than the data support, and a positive value "
        "less. The map was calculated about PDB-Redo's model, not the "
        "deposited one, so where pdb_redo_revised shows the two differ at "
        "the atom, density_fofc_sigma_pdb_redo is the value about the model "
        f"the map belongs to. Not thresholded. {_MISSING} Also missing when "
        "the deposited position is not a finite point."
    ),
    "density_2fofc_sigma_pdb_redo": (
        "Density of PDB-Redo's 2mFo−DFc map at the position PDB-Redo's "
        f"re-refined model gives the atom, {_SIGMA}. {_MISSING} Also missing "
        "for an atom with no partner in PDB-Redo's model, such as a water or "
        "a hydrogen PDB-Redo did not keep."
    ),
    "density_fofc_sigma_pdb_redo": (
        "Density of PDB-Redo's mFo−DFc difference map at the position "
        f"PDB-Redo's re-refined model gives the atom, {_SIGMA}. This is the "
        "difference density at an atom of the model the map was calculated "
        "about. A negative value says that model places more density at its "
        "atom than the data support, as at an atom that is partly absent "
        f"from the crystal. Not thresholded. {_MISSING} Also missing for an "
        "atom with no partner in PDB-Redo's model."
    ),
    "density_fofc_eA3": (
        "Difference density at the atom's deposited position on the scale of "
        "density_fc_eA3, nominally e/Å³. It is the density of PDB-Redo's "
        "2mFo−DFc map less that of its calculated map at the point, which is "
        "the density of a map of 2(mFo−DFc). Divided by density_fc_eA3 it is "
        "the share of the calculated density at the point that the data do "
        "not support: negative where the model places too much, and −1 "
        "where the 2mFo−DFc density is at the mean of the unit cell. The "
        "engine writes no such ratio, since it is unstable where the "
        "calculated density is small. PDB-Redo's own difference coefficients "
        "are 2(mFo−DFc), and half of that in a twinned refinement, so the "
        "value equals density_fofc_sigma times density_fofc_rms for an "
        "untwinned entry and twice that product for a twinned one. Not "
        f"thresholded. {_MISSING} {_NO_CALCULATED}, and when the deposited "
        "position is not a finite point."
    ),
    "density_fc_eA3": (
        "Density of the map of PDB-Redo's calculated structure factors "
        f"({MAP_FC_LABELS[0]}, {MAP_FC_LABELS[1]}) at the atom's deposited "
        "position, on the scale of the map coefficients, nominally e/Å³, "
        "with the map's mean subtracted. It is what PDB-Redo's whole model, "
        "bulk solvent included, places at the point at the resolution of the "
        "data, and it is the DFc of the two other maps. The map has no "
        "F(000) term, so the value is the density above the mean of the unit "
        "cell and not above zero. Every atom near the point adds to it, so "
        "it is not the atom's own share. It falls with the atom's B-factor "
        "and occupancy and with coarser data, and can be near zero or "
        "negative. The coefficients of a twinned refinement are on a scale "
        "of their own, so compare the value between entries only for "
        "untwinned ones. Read by cubic interpolation, and not thresholded. "
        f"{_MISSING} {_NO_CALCULATED}, and when the deposited position is "
        "not a finite point."
    ),
    "density_fofc_eA3_pdb_redo": (
        "Difference density at the position PDB-Redo's re-refined model "
        "gives the atom, on the scale of density_fc_eA3_pdb_redo and "
        "defined as density_fofc_eA3 is. Divided by density_fc_eA3_pdb_redo "
        "it is the share of the density PDB-Redo's model places at its own "
        "atom that the data do not support. Unlike a value in sigma, that "
        "share does not grow with the number of electrons of the element. "
        f"Not thresholded. {_MISSING} {_NO_CALCULATED}, and for an atom with "
        "no partner in PDB-Redo's model."
    ),
    "density_fc_eA3_pdb_redo": (
        "Density of the map of PDB-Redo's calculated structure factors at the "
        "position PDB-Redo's re-refined model gives the atom, on the scale "
        "and under the cautions of density_fc_eA3. This is the calculated "
        "density at an atom of the model the map was calculated from. "
        f"{_MISSING} {_NO_CALCULATED}, and for an atom with no partner in "
        "PDB-Redo's model."
    ),
    "density_status": (
        f"{ATOM_STATUS_OK} when both positions were read. "
        f"{ATOM_STATUS_NO_PDB_REDO_ATOM} when PDB-Redo's model has no partner "
        "for the atom, so only the deposited position was read. "
        f"{ATOM_STATUS_NO_COORDINATES} when the deposited position is not a "
        "finite point, so the deposited values are missing. For an entry "
        "without maps it holds the entry's density_map_status on every atom."
    ),
    "density_2fofc_rms": (
        "Root-mean-square deviation of PDB-Redo's 2mFo−DFc map over the unit "
        "cell, on the scale of the map coefficients, nominally e/Å³. "
        "Multiplying a density_2fofc_sigma value by it puts the value back on "
        "that scale. Missing when the entry has no maps."
    ),
    "density_fofc_rms": (
        "Root-mean-square deviation of PDB-Redo's mFo−DFc map over the unit "
        "cell, on the scale of the map coefficients, nominally e/Å³. "
        "Multiplying a density_fofc_sigma value by it puts the value back on "
        "that scale, which depends less on the entry's resolution and "
        "solvent content than the sigma scale does. Missing when the entry "
        "has no maps."
    ),
    "density_map_resolution_A": (
        "High-resolution limit in Å of the reflections in PDB-Redo's "
        "reflection file, which the maps were made from. Missing when the "
        "entry has no maps."
    ),
    "density_map_sample_rate": (
        "Grid points per high-resolution limit the maps were sampled at, "
        f"{MAP_SAMPLE_RATE:g} unless the unit cell would then need more than "
        f"{MAP_MAX_GRID_POINTS:,} grid points, and never below "
        f"{MAP_MIN_SAMPLE_RATE:g}. A lower rate reads a value less exactly. "
        "Missing when the entry has no maps."
    ),
    "density_map_status": (
        f"{MAP_STATUS_OK} when both maps were made and read. "
        f"{MAP_STATUS_NO_ENTRY} when PDB-Redo has no reflection file for the "
        f"entry. {MAP_STATUS_NO_COEFFICIENTS} when the file lacks a map "
        "coefficient column or gives a map without usable density. "
        f"{MAP_STATUS_TOO_LARGE} when the unit cell needs more grid points "
        f"than the limit at the lowest sample rate. {MAP_STATUS_CELL_MISMATCH} "
        "when the reflection file's unit cell is not the deposited one. "
        f"{MAP_STATUS_NOT_PAIRED} when no atom paired with PDB-Redo's model, "
        "so nothing shows that the deposited model and the maps share a "
        "frame. Missing when the engine failed."
    ),
}

_GRADIENT = (
    "in sigma per Å, the map's root-mean-square density over the unit cell "
    "per ångström. It is the slope of the interpolated map at the point, and "
    "the three components are a vector that points where the difference "
    "density rises, which is where the data would move the atom. An atom "
    "that restraints hold short of where the data put it sits between a "
    "negative lobe on one side and a positive lobe on the other, and the "
    "vector points from the first to the second, while the value at the atom "
    "can be near zero. Projected on a bond, it says whether the data want "
    "the bond longer or shorter. Like a value in sigma it grows with the "
    "atom's electrons for the same displacement, so compare it within one "
    "element. Not thresholded"
)

DESCRIPTIONS.update(
    {
        name: (
            f"Gradient of PDB-Redo's mFo−DFc difference map along the {axis} "
            "axis of the model's frame, at the position PDB-Redo's re-refined "
            f"model gives the atom, {_GRADIENT}. {_MISSING} Also missing for "
            "an atom with no partner in PDB-Redo's model."
        )
        for axis, name in zip("xyz", PDB_REDO_GRADIENT_COLUMNS, strict=True)
    }
)

DESCRIPTIONS.update(
    {
        name: (
            f"Shift in Å along the {axis} axis of the model's frame that would "
            "level the slope of the difference density at the position "
            "PDB-Redo's re-refined model gives the atom, to first order. It is "
            "minus the inverse of the calculated map's curvature at the point "
            "times the gradient of the difference density, both on the scale "
            "of density_fofc_eA3_pdb_redo, so it does not grow with the "
            "atom's electrons and does not depend on the scale of a twinned "
            "refinement. The curvature is that of PDB-Redo's whole model, so "
            "a bonded neighbour flattens it along the bond, and a neighbour "
            "that moves too adds to the slope. Both grow at coarser "
            "resolution: with disulfide sulfurs moved apart in noiseless "
            "data, the shift along the bond was 1.00 to 1.08 times the true "
            "one at 1.25 Å or finer and 1.35 to 2.6 times it at 1.9 to "
            "2.3 Å. So compare shifts within bins of resolution. Where the "
            "calculated map curves downward only slightly in one direction "
            "the shift along it is large, so summarize shifts by medians. It "
            "estimates how far the data would move the atom and is not a "
            f"refined position. Not thresholded. {_MISSING} "
            f"{_NO_CALCULATED}, for an atom with no partner in PDB-Redo's "
            "model, and where the calculated map does not curve downward in "
            "every direction at the point, so that no shift levels the slope, "
            "as at a fifth of carbonyl carbons, whose density is often a "
            "shoulder of their oxygen's."
        )
        for axis, name in zip("xyz", PDB_REDO_SHIFT_COLUMNS, strict=True)
    }
)

_WWPDB_MISSING = (
    "Missing when the entry has no wwPDB validation maps, and "
    "density_status_wwpdb then says why."
)

DESCRIPTIONS.update(
    {
        "density_2fofc_sigma_wwpdb": (
            "Density of the wwPDB validation 2mFo−DFc map at the atom's "
            f"deposited position, {_SIGMA}. The map was calculated about the "
            "deposited model, the one the atom table holds, with no further "
            f"refinement. Not thresholded. {_WWPDB_MISSING} Also missing when "
            "the deposited position is not a finite point."
        ),
        "density_fofc_sigma_wwpdb": (
            "Density of the wwPDB validation mFo−DFc difference map at the "
            f"atom's deposited position, {_SIGMA}. A negative value says the "
            "deposited model places more density at its atom than the data "
            "support, as at an atom that is partly absent from the crystal. "
            f"Not thresholded. {_WWPDB_MISSING} Also missing when the "
            "deposited position is not a finite point."
        ),
        **{
            name: (
                "Gradient of the wwPDB validation mFo−DFc difference map along "
                f"the {axis} axis of the model's frame, at the atom's deposited "
                f"position, {_GRADIENT}. The map was calculated about the "
                "deposited model, so the vector points where the data would "
                f"move the depositor's atom. {_WWPDB_MISSING} Also missing "
                "when the deposited position is not a finite point."
            )
            for axis, name in zip("xyz", WWPDB_GRADIENT_COLUMNS, strict=True)
        },
        "density_status_wwpdb": (
            f"{ATOM_STATUS_OK} when the atom's position was read in the wwPDB "
            f"validation maps. {ATOM_STATUS_NO_COORDINATES} when the deposited "
            "position is not a finite point. For an entry without such maps "
            "it holds the entry's density_map_status_wwpdb on every atom."
        ),
        "density_2fofc_rms_wwpdb": (
            "Root-mean-square deviation of the wwPDB validation 2mFo−DFc map "
            "over the unit cell, on the scale of the map coefficients. Missing "
            "when the entry has no such maps."
        ),
        "density_fofc_rms_wwpdb": (
            "Root-mean-square deviation of the wwPDB validation mFo−DFc map "
            "over the unit cell, on the scale of the map coefficients. Missing "
            "when the entry has no such maps."
        ),
        "density_map_resolution_A_wwpdb": (
            "High-resolution limit in Å of the reflections in the wwPDB "
            "validation map files. Missing when the entry has no such maps."
        ),
        "density_map_sample_rate_wwpdb": (
            "Grid points per high-resolution limit the wwPDB validation maps "
            "were sampled at, under the same rule as density_map_sample_rate. "
            "Missing when the entry has no such maps."
        ),
        "density_map_status_wwpdb": (
            f"{MAP_STATUS_OK} when both wwPDB validation maps were made and "
            f"read. {MAP_STATUS_NO_VALIDATION_MAPS} when the archive has no "
            "validation map coefficients for the entry, as for an entry "
            f"deposited without structure factors. {MAP_STATUS_NO_COEFFICIENTS} "
            "when a file lacks its coefficients or gives a map without usable "
            f"density. {MAP_STATUS_TOO_LARGE} when the unit cell needs more "
            "grid points than the limit at the lowest sample rate. "
            f"{MAP_STATUS_CELL_MISMATCH} when a file's unit cell is not the "
            "deposited one. Missing when the engine failed."
        ),
    }
)


def _reads_pdb_redo(options: EngineOptions) -> bool:
    return density_source_selected(options, DENSITY_SOURCE_PDB_REDO)


def _reads_wwpdb(options: EngineOptions) -> bool:
    return density_source_selected(options, DENSITY_SOURCE_WWPDB)


DENSITY_COLUMNS: tuple[ColumnSpec, ...] = tuple(
    ColumnSpec(
        name=name,
        group=GROUP,
        # Each family is written only when the run reads its maps.
        enabled_if=_reads_wwpdb if name in WWPDB_COLUMN_NAMES else _reads_pdb_redo,
        granularity=(
            ColumnGranularity.STRUCTURE
            if name in _STRUCTURE_COLUMNS
            else ColumnGranularity.ATOM
        ),
        dtype="text" if name in _TEXT_COLUMNS else "numeric",
        description=DESCRIPTIONS[name],
        # The maps cost a download per entry, so only the profile made for
        # radiation-damage features switches the engine on.
        profiles=frozenset({PROFILE_RADIATION_SENSITIVITY}),
    )
    for name in (*PDB_REDO_COLUMN_NAMES, *WWPDB_COLUMN_NAMES)
)

__all__ = [
    "ATOM_STATUS_COLUMN",
    "DENSITY_COLUMNS",
    "DEPOSITED_ABSOLUTE_COLUMNS",
    "DEPOSITED_VALUE_COLUMNS",
    "DESCRIPTIONS",
    "GROUP",
    "MAP_RESOLUTION_COLUMN",
    "MAP_SAMPLE_RATE_COLUMN",
    "MAP_STATUS_COLUMN",
    "PDB_REDO_ABSOLUTE_COLUMNS",
    "PDB_REDO_COLUMN_NAMES",
    "PDB_REDO_GRADIENT_COLUMNS",
    "PDB_REDO_SHIFT_COLUMNS",
    "PDB_REDO_VALUE_COLUMNS",
    "RMS_COLUMNS",
    "WWPDB_ATOM_STATUS_COLUMN",
    "WWPDB_COLUMN_NAMES",
    "WWPDB_GRADIENT_COLUMNS",
    "WWPDB_MAP_RESOLUTION_COLUMN",
    "WWPDB_MAP_SAMPLE_RATE_COLUMN",
    "WWPDB_MAP_STATUS_COLUMN",
    "WWPDB_RMS_COLUMNS",
    "WWPDB_SUFFIX",
    "WWPDB_VALUE_COLUMNS",
]
