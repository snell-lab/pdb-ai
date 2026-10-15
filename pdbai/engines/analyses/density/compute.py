"""Reading an entry's maps at the atoms of its structure.

compute_structure_density takes the atom table of one structure and the path
of PDB-Redo's reflection file, and returns every column of PDB-Redo's maps on
the atom table's index. compute_validation_density does the same for the two
wwPDB validation map files, which were calculated about the deposited model
and are read at the deposited position alone. Nothing here downloads, and
nothing judges a value.

The two maps are made one after the other from the map coefficients, each as
a grid over the whole unit cell, and each is read by cubic interpolation.
A value is the map's density at the point less the map's mean, divided by
its root-mean-square deviation over the unit cell, which is the scale map
viewers call sigma. The mean of a map made without the F(000) term is zero
to rounding, and it is subtracted all the same.

PDB-Redo's file also holds the calculated structure factors of its model,
and their map is made third and read at the same points. It is left on the
scale of the coefficients, less its mean, as the density the model places
at an atom. The 2mFo−DFc density less the calculated density, on that same
scale, is the difference density beside it. The file's own difference
coefficients are that difference at every reflection, or half of it in a
twinned refinement, so taking the difference of the two readings puts every
entry on one scale. The wwPDB validation files hold no calculated structure
factors, so they give neither value.

Each map is read at two positions for every atom. One is the atom's own
position in the atom table, which is the deposited model's. The other is the
position PDB-Redo's model gives the same atom, which the PDB-Redo comparison
columns pdb_redo_x, pdb_redo_y, and pdb_redo_z hold. The maps describe
PDB-Redo's model, so the second position is the one the difference map was
calculated about, and the first says what the same data show where the
depositor put the atom.

The gradient of the difference map is read only at the atoms of the model
the map was calculated about: PDB-Redo's position for PDB-Redo's maps, and
the deposited one for the wwPDB maps. It is read from a map of the
derivative itself, whose coefficients are those of the map times the
reciprocal vector, so it is exact at the grid points and read between them
by the same cubic interpolation as the values. From PDB-Redo's file the
gradient of the 2(mFo−DFc) density and the curvature of the calculated
density, from six maps of second derivatives, give the shift of the atom
that would level the slope. The first gradient is the difference map's own
times the factor between the two difference densities, 1 or 2, which the
values read at the atoms give. Each derivative map is made, read, and let go
before the next, so one map is still held at a time.

Three checks keep a value from being read in the wrong place. A reflection
file whose cell is not the deposited cell gives no values. Neither does an
entry in which no atom paired with PDB-Redo's model, since the pairing is the
only evidence that the two models share one frame. And an atom without
finite coordinates gives none at that position.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import gemmi
import numpy as np
import numpy.typing as npt
import pandas as pd

from .columns import (
    ATOM_STATUS_COLUMN,
    DEPOSITED_ABSOLUTE_COLUMNS,
    DEPOSITED_VALUE_COLUMNS,
    MAP_RESOLUTION_COLUMN,
    MAP_SAMPLE_RATE_COLUMN,
    MAP_STATUS_COLUMN,
    PDB_REDO_ABSOLUTE_COLUMNS,
    PDB_REDO_GRADIENT_COLUMNS,
    PDB_REDO_SHIFT_COLUMNS,
    PDB_REDO_VALUE_COLUMNS,
    RMS_COLUMNS,
    WWPDB_ATOM_STATUS_COLUMN,
    WWPDB_GRADIENT_COLUMNS,
    WWPDB_MAP_RESOLUTION_COLUMN,
    WWPDB_MAP_SAMPLE_RATE_COLUMN,
    WWPDB_MAP_STATUS_COLUMN,
    WWPDB_RMS_COLUMNS,
    WWPDB_VALUE_COLUMNS,
)
from .constants import (
    ATOM_STATUS_NO_COORDINATES,
    ATOM_STATUS_NO_PDB_REDO_ATOM,
    ATOM_STATUS_OK,
    CELL_ANGLE_TOLERANCE_DEG,
    CELL_LENGTH_TOLERANCE,
    MAP_2FOFC_LABELS,
    MAP_FC_LABELS,
    MAP_FOFC_LABELS,
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
    WWPDB_2FOFC_LABELS,
    WWPDB_FOFC_LABELS,
)

_DEPOSITED_POSITION_COLUMNS = ("x", "y", "z")
_PDB_REDO_POSITION_COLUMNS = ("pdb_redo_x", "pdb_redo_y", "pdb_redo_z")

# The two maps in the order of the value columns.
_MAPS = (MAP_2FOFC_LABELS, MAP_FOFC_LABELS)

type Cell = tuple[float, float, float, float, float, float]
# A set of reflections gemmi can transform to a map, PDB-Redo's MTZ file or a
# reflection block of a wwPDB validation file.
type Reflections = gemmi.Mtz | gemmi.ReflnBlock


@dataclass(frozen=True)
class DensityComputation:
    """The density columns of one structure, and how its maps turned out."""

    # Every density column, on the atom table's index.
    columns: pd.DataFrame
    map_status: str
    # The rate the maps were sampled at, or None when none was made.
    sample_rate: float | None
    warnings: tuple[str, ...]


def _positions(atoms: pd.DataFrame, names: Sequence[str]) -> npt.NDArray[np.float64]:
    """Return the named coordinate columns as an array with a row per atom.

    A column the atoms lack, and a value that is not a number, give NaN.
    """
    columns = [
        pd.to_numeric(atoms[name], errors="coerce").to_numpy(
            dtype=np.float64, na_value=np.nan
        )
        if name in atoms.columns
        else np.full(len(atoms), np.nan)
        for name in names
    ]
    return np.column_stack(columns) if columns else np.empty((len(atoms), 0))


def _empty_columns(index: pd.Index, status: str) -> pd.DataFrame:
    """Return the density columns of a structure without maps."""
    columns = pd.DataFrame(
        {
            name: np.full(len(index), np.nan)
            for name in (
                *DEPOSITED_VALUE_COLUMNS,
                *PDB_REDO_VALUE_COLUMNS,
                *DEPOSITED_ABSOLUTE_COLUMNS,
                *PDB_REDO_ABSOLUTE_COLUMNS,
                *PDB_REDO_GRADIENT_COLUMNS,
                *PDB_REDO_SHIFT_COLUMNS,
                *RMS_COLUMNS,
                MAP_RESOLUTION_COLUMN,
                MAP_SAMPLE_RATE_COLUMN,
            )
        },
        index=index,
    )
    columns[ATOM_STATUS_COLUMN] = pd.Series(status, index=index, dtype=object)
    columns[MAP_STATUS_COLUMN] = pd.Series(status, index=index, dtype=object)
    return columns[list(_COLUMN_ORDER)]


_COLUMN_ORDER = (
    *DEPOSITED_VALUE_COLUMNS,
    *PDB_REDO_VALUE_COLUMNS,
    *DEPOSITED_ABSOLUTE_COLUMNS,
    *PDB_REDO_ABSOLUTE_COLUMNS,
    *PDB_REDO_GRADIENT_COLUMNS,
    *PDB_REDO_SHIFT_COLUMNS,
    ATOM_STATUS_COLUMN,
    *RMS_COLUMNS,
    MAP_RESOLUTION_COLUMN,
    MAP_SAMPLE_RATE_COLUMN,
    MAP_STATUS_COLUMN,
)


def _without_maps(
    index: pd.Index, status: str, warning: str | None = None
) -> DensityComputation:
    return DensityComputation(
        columns=_empty_columns(index, status),
        map_status=status,
        sample_rate=None,
        warnings=() if warning is None else (warning,),
    )


def cells_agree(map_cell: Cell, deposited_cell: Cell) -> bool:
    """Return whether two unit cells describe the same crystal setting.

    Each length must agree within CELL_LENGTH_TOLERANCE of the deposited one,
    and each angle within CELL_ANGLE_TOLERANCE_DEG.
    """
    lengths = all(
        abs(ours - theirs) <= CELL_LENGTH_TOLERANCE * theirs
        for ours, theirs in zip(map_cell[:3], deposited_cell[:3], strict=True)
    )
    angles = all(
        abs(ours - theirs) <= CELL_ANGLE_TOLERANCE_DEG
        for ours, theirs in zip(map_cell[3:], deposited_cell[3:], strict=True)
    )
    return lengths and angles


def choose_sample_rate(mtz: Reflections) -> float | None:
    """Return the rate to sample the entry's maps at, or None when too large.

    The rate is MAP_SAMPLE_RATE when the grid then holds at most
    MAP_MAX_GRID_POINTS, and otherwise the highest rate, to one decimal, that
    keeps it within the limit. None stands for a unit cell that exceeds the
    limit even at MAP_MIN_SAMPLE_RATE.
    """
    rate = MAP_SAMPLE_RATE
    while rate >= MAP_MIN_SAMPLE_RATE - 1e-9:
        size = mtz.get_size_for_hkl(sample_rate=rate)
        if math.prod(size) <= MAP_MAX_GRID_POINTS:
            return rate
        rate = round(rate - 0.1, 1)
    return None


def _read_map(
    mtz: Reflections,
    labels: tuple[str, str],
    sample_rate: float,
    positions: Sequence[npt.NDArray[np.float64]],
) -> tuple[float, list[npt.NDArray[np.float64]]] | None:
    """Return a map's root-mean-square density and its values at each position set.

    A value is the density at the point less the map's mean, on the scale of
    the coefficients, and dividing it by the root-mean-square density gives
    it in sigma. Each array of ``positions`` holds a row per point, and a row
    with a NaN gives NaN. None stands for a map that has no variation or
    holds a value that is not finite, which a file without usable
    coefficients gives.
    """
    grid = mtz.transform_f_phi_to_map(labels[0], labels[1], sample_rate=sample_rate)
    density = np.asarray(grid)
    mean = float(density.mean(dtype=np.float64))
    rms = float(density.std(dtype=np.float64))
    if not (math.isfinite(mean) and math.isfinite(rms)) or rms <= 0.0:
        return None
    values: list[npt.NDArray[np.float64]] = []
    for points in positions:
        result = np.full(len(points), np.nan)
        finite = np.isfinite(points).all(axis=1)
        if finite.any():
            raw = grid.interpolate_position_array(
                np.ascontiguousarray(points[finite]), order=3
            )
            result[finite] = np.asarray(raw, dtype=np.float64) - mean
        values.append(result)
    return rms, values


# The pairs of orthogonal axes of the six distinct components of a curvature.
_CURVATURE_AXES = ((0, 0), (1, 1), (2, 2), (0, 1), (0, 2), (1, 2))


def _derivative_map(
    reflections: Reflections,
    labels: tuple[str, str],
    sample_rate: float,
    axes: tuple[int, ...],
) -> gemmi.FloatGrid:
    """Return the map of a derivative of a map along orthogonal axes.

    One axis gives a component of the gradient, and two give a component of
    the curvature. The coefficients are put over the whole of reciprocal
    space, which gemmi fills from the asymmetric unit by the space group's
    symmetry, and each is multiplied by −2πi times its reciprocal vector's
    component along each axis named. That differentiates the Fourier series
    term by term, so the map is the exact derivative of the density at its
    grid points, on the scale of the coefficients per Å for each axis. The
    grid is that of the density at the same sample rate.
    """
    size = reflections.get_size_for_hkl(sample_rate=sample_rate)
    grid = reflections.get_f_phi_on_grid(labels[0], labels[1], size, half_l=True)
    coefficients = np.asarray(grid)
    # The Miller indices of the grid, the first two wrapping to negative
    # values past half the grid and the third running over half of it.
    h_values = _wrapped_indices(coefficients.shape[0])
    k_values = _wrapped_indices(coefficients.shape[1])
    l_values = np.arange(coefficients.shape[2], dtype=np.float64)
    # A reciprocal vector in the model's frame is the transpose of the
    # fractionalization matrix times its indices.
    fractionalization = np.array(grid.unit_cell.frac.mat.tolist(), dtype=np.float64)
    # One plane of constant h at a time, so that no array of the grid's size
    # is made beside the coefficients.
    for h_index, h in enumerate(h_values):
        factor = np.ones((len(k_values), len(l_values)), dtype=np.complex128)
        for axis in axes:
            component = (
                fractionalization[0, axis] * h
                + fractionalization[1, axis] * k_values[:, None]
                + fractionalization[2, axis] * l_values[None, :]
            )
            factor *= -2j * math.pi * component
        coefficients[h_index] *= factor.astype(np.complex64)
    return gemmi.transform_f_phi_grid_to_map(grid)


def _wrapped_indices(size: int) -> npt.NDArray[np.float64]:
    """Return the Miller index of each grid point along an axis of ``size`` points.

    The indices run up from zero and wrap to negative values past half the
    grid, as ``numpy.fft.fftfreq(size, 1 / size)`` gives them.
    """
    indices = np.arange(size, dtype=np.float64)
    indices[indices >= (size + 1) // 2] -= size
    return indices


def _read_derivatives(
    reflections: Reflections,
    labels: tuple[str, str],
    sample_rate: float,
    points: npt.NDArray[np.float64],
    components: Sequence[tuple[int, ...]],
) -> npt.NDArray[np.float64]:
    """Return derivatives of a map at each point, a column per component.

    Each component names the axes of one derivative, as _derivative_map
    takes them, and its map is made, read by cubic interpolation, and let go
    before the next is made. A row with a NaN gives NaN.
    """
    result = np.full((len(points), len(components)), np.nan)
    finite = np.isfinite(points).all(axis=1)
    if not finite.any():
        return result
    located = np.ascontiguousarray(points[finite])
    for column, axes in enumerate(components):
        grid = _derivative_map(reflections, labels, sample_rate, axes)
        result[finite, column] = np.asarray(
            grid.interpolate_position_array(located, order=3), dtype=np.float64
        )
        # Let the map go before the next is made, or two are held at once.
        del grid
    return result


def _read_gradients(
    reflections: Reflections,
    labels: tuple[str, str],
    sample_rate: float,
    points: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    """Return the gradient of a map at each point, a row of x, y, and z."""
    return _read_derivatives(
        reflections, labels, sample_rate, points, ((0,), (1,), (2,))
    )


def _read_curvatures(
    reflections: Reflections,
    labels: tuple[str, str],
    sample_rate: float,
    points: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    """Return the curvature of a map at each point, a 3 by 3 matrix per point."""
    components = _read_derivatives(
        reflections, labels, sample_rate, points, _CURVATURE_AXES
    )
    result = np.empty((len(points), 3, 3))
    for column, (first, second) in enumerate(_CURVATURE_AXES):
        result[:, first, second] = components[:, column]
        result[:, second, first] = components[:, column]
    return result


def _difference_scale(
    own: npt.NDArray[np.float64], taken: npt.NDArray[np.float64]
) -> float:
    """Return the factor from the file's difference density to the 2(mFo−DFc) one.

    ``own`` is the density of the file's difference map at a set of points,
    and ``taken`` the 2mFo−DFc density less the calculated density at the
    same points, both on the scale of the coefficients. The factor is that of
    least squares through zero over the points where both are finite. It is
    1, or 2 for a twinned refinement, whose difference coefficients are half
    of FWT less FC_ALL. NaN stands for no such point.
    """
    both = np.isfinite(own) & np.isfinite(taken)
    weight = float(np.dot(own[both], own[both]))
    return float(np.dot(own[both], taken[both])) / weight if weight > 0 else math.nan


def level_shifts(
    slopes: npt.NDArray[np.float64], curvatures: npt.NDArray[np.float64]
) -> npt.NDArray[np.float64]:
    """Return the shift that levels each slope, a row of x, y, and z per point.

    ``slopes`` is the gradient of the difference density at each point and
    ``curvatures`` the curvature of the calculated density there, on one
    scale. The shift is minus the inverse curvature times the slope, the
    step of Newton's method toward the point where the slope is zero. A
    point where the curvature is not negative in every direction has no such
    point nearby and gives NaN, as does a point with a NaN.
    """
    result = np.full((len(slopes), 3), np.nan)
    finite = np.isfinite(slopes).all(axis=1) & np.isfinite(curvatures).all(axis=(1, 2))
    rows = np.flatnonzero(finite)
    if len(rows):
        rows = rows[np.linalg.eigvalsh(curvatures[rows])[:, -1] < 0.0]
    if len(rows):
        result[rows] = -np.linalg.solve(curvatures[rows], slopes[rows][:, :, None])[
            :, :, 0
        ]
    return result


def compute_structure_density(
    atoms: pd.DataFrame,
    mtz_path: Path | None,
    *,
    deposited_cell: Cell | None = None,
) -> DensityComputation:
    """Return the density columns of one structure.

    ``atoms`` is the structure's atom table with the PDB-Redo comparison
    columns, and ``mtz_path`` the path of PDB-Redo's reflection file, or None
    when PDB-Redo has none for the entry. ``deposited_cell`` is the cell of
    the deposited model, as three lengths in Å and three angles in degrees,
    and the cell check is skipped without it.

    An entry without maps gets empty values and the reason as both statuses,
    with a warning. A file with the two maps and no calculated structure
    factors keeps their values and leaves the calculated density and the
    difference density on its scale empty, with a warning. Raises what
    reading the reflection file raises.
    """
    index = atoms.index
    if mtz_path is None:
        return _without_maps(
            index,
            MAP_STATUS_NO_ENTRY,
            "PDB-Redo has no reflection file for this entry, so its density "
            "columns are empty.",
        )
    mtz = gemmi.read_mtz_file(str(mtz_path))
    labels = {column.label for column in mtz.columns}
    missing = [label for pair in _MAPS for label in pair if label not in labels]
    if missing or mtz.nreflections == 0:
        return _without_maps(
            index,
            MAP_STATUS_NO_COEFFICIENTS,
            "PDB-Redo's reflection file lacks the map coefficient column(s) "
            + (", ".join(missing) if missing else "of any reflection")
            + ", so the density columns are empty.",
        )
    cell = mtz.cell
    map_cell: Cell = (cell.a, cell.b, cell.c, cell.alpha, cell.beta, cell.gamma)
    if deposited_cell is not None and not cells_agree(map_cell, deposited_cell):
        return _without_maps(
            index,
            MAP_STATUS_CELL_MISMATCH,
            "The unit cell of PDB-Redo's reflection file "
            f"({', '.join(f'{value:g}' for value in map_cell)}) is not the "
            "deposited cell "
            f"({', '.join(f'{value:g}' for value in deposited_cell)}), so the "
            "density columns are empty.",
        )
    pdb_redo_positions = _positions(atoms, _PDB_REDO_POSITION_COLUMNS)
    paired = np.isfinite(pdb_redo_positions).all(axis=1)
    if not paired.any():
        return _without_maps(
            index,
            MAP_STATUS_NOT_PAIRED,
            "No atom paired with PDB-Redo's model, so nothing shows that the "
            "deposited model and PDB-Redo's maps share a frame, and the density "
            "columns are empty.",
        )
    sample_rate = choose_sample_rate(mtz)
    if sample_rate is None:
        return _without_maps(
            index,
            MAP_STATUS_TOO_LARGE,
            f"The maps would exceed {MAP_MAX_GRID_POINTS:,} grid points at a "
            f"sample rate of {MAP_MIN_SAMPLE_RATE:g}, so the density columns "
            "are empty.",
        )
    deposited_positions = _positions(atoms, _DEPOSITED_POSITION_COLUMNS)
    located = np.isfinite(deposited_positions).all(axis=1)

    columns = _empty_columns(index, MAP_STATUS_OK)
    position_sets = (deposited_positions, pdb_redo_positions)
    # Each map's density at the two position sets, on the coefficient scale.
    readings: list[list[npt.NDArray[np.float64]]] = []
    rms_by_map: list[float] = []
    for labels_of_map, deposited_name, pdb_redo_name, rms_name in zip(
        _MAPS, DEPOSITED_VALUE_COLUMNS, PDB_REDO_VALUE_COLUMNS, RMS_COLUMNS, strict=True
    ):
        read = _read_map(mtz, labels_of_map, sample_rate, position_sets)
        if read is None:
            return _without_maps(
                index,
                MAP_STATUS_NO_COEFFICIENTS,
                f"The {labels_of_map[0]} map of PDB-Redo's reflection file has no "
                "usable density, so the density columns are empty.",
            )
        rms, values = read
        at_deposited, at_pdb_redo = values
        columns[deposited_name] = at_deposited / rms
        columns[pdb_redo_name] = at_pdb_redo / rms
        columns[rms_name] = rms
        rms_by_map.append(rms)
        readings.append(values)
    # The difference map was the second. Its gradient is read at the atoms of
    # the model it was calculated about.
    gradients = _read_gradients(mtz, MAP_FOFC_LABELS, sample_rate, pdb_redo_positions)
    columns[list(PDB_REDO_GRADIENT_COLUMNS)] = gradients / rms_by_map[1]
    warnings: list[str] = []
    # The calculated map is made third, so one map is still held at a time.
    # A file without it keeps the values of the two other maps.
    missing_calculated = [label for label in MAP_FC_LABELS if label not in labels]
    calculated = (
        None
        if missing_calculated
        else _read_map(mtz, MAP_FC_LABELS, sample_rate, position_sets)
    )
    if calculated is None:
        warnings.append(
            "PDB-Redo's reflection file "
            + (
                "lacks the calculated structure factor column(s) "
                + ", ".join(missing_calculated)
                if missing_calculated
                else f"gives a {MAP_FC_LABELS[0]} map without usable density"
            )
            + ", so the calculated density, the difference density on its "
            "scale, and the shifts are empty."
        )
    else:
        # The 2mFo−DFc map is the first of the two, and its density less the
        # calculated one is the density of the 2(mFo−DFc) map at the point.
        for (difference_name, calculated_name), observed, model in zip(
            (DEPOSITED_ABSOLUTE_COLUMNS, PDB_REDO_ABSOLUTE_COLUMNS),
            readings[0],
            calculated[1],
            strict=True,
        ):
            columns[difference_name] = observed - model
            columns[calculated_name] = model
        # The slope of the 2(mFo−DFc) density over the curvature of the
        # calculated density, both on the scale of the coefficients. The
        # first is the difference map's own slope times the factor between
        # the two difference densities, so no map of it is made.
        curvatures = _read_curvatures(
            mtz, MAP_FC_LABELS, sample_rate, pdb_redo_positions
        )
        scale = _difference_scale(
            np.concatenate(readings[1]),
            np.concatenate(readings[0]) - np.concatenate(calculated[1]),
        )
        columns[list(PDB_REDO_SHIFT_COLUMNS)] = level_shifts(
            gradients * scale, curvatures
        )
    columns[MAP_RESOLUTION_COLUMN] = float(mtz.resolution_high())
    columns[MAP_SAMPLE_RATE_COLUMN] = sample_rate
    status = np.full(len(index), ATOM_STATUS_OK, dtype=object)
    status[~paired] = ATOM_STATUS_NO_PDB_REDO_ATOM
    status[~located] = ATOM_STATUS_NO_COORDINATES
    columns[ATOM_STATUS_COLUMN] = status
    if sample_rate < MAP_SAMPLE_RATE:
        warnings.append(
            f"The maps were sampled at a rate of {sample_rate:g} in place of "
            f"{MAP_SAMPLE_RATE:g} to stay within {MAP_MAX_GRID_POINTS:,} grid "
            "points."
        )
    return DensityComputation(
        columns=columns,
        map_status=MAP_STATUS_OK,
        sample_rate=sample_rate,
        warnings=tuple(warnings),
    )


def _empty_validation_columns(index: pd.Index, status: str) -> pd.DataFrame:
    """Return the wwPDB density columns of a structure without such maps."""
    columns = pd.DataFrame(
        {
            name: np.full(len(index), np.nan)
            for name in (
                *WWPDB_VALUE_COLUMNS,
                *WWPDB_GRADIENT_COLUMNS,
                *WWPDB_RMS_COLUMNS,
                WWPDB_MAP_RESOLUTION_COLUMN,
                WWPDB_MAP_SAMPLE_RATE_COLUMN,
            )
        },
        index=index,
    )
    columns[WWPDB_ATOM_STATUS_COLUMN] = pd.Series(status, index=index, dtype=object)
    columns[WWPDB_MAP_STATUS_COLUMN] = pd.Series(status, index=index, dtype=object)
    return columns[
        [
            *WWPDB_VALUE_COLUMNS,
            *WWPDB_GRADIENT_COLUMNS,
            WWPDB_ATOM_STATUS_COLUMN,
            *WWPDB_RMS_COLUMNS,
            WWPDB_MAP_RESOLUTION_COLUMN,
            WWPDB_MAP_SAMPLE_RATE_COLUMN,
            WWPDB_MAP_STATUS_COLUMN,
        ]
    ]


def _without_validation_maps(
    index: pd.Index, status: str, warning: str
) -> DensityComputation:
    return DensityComputation(
        columns=_empty_validation_columns(index, status),
        map_status=status,
        sample_rate=None,
        warnings=(warning,),
    )


def _read_reflection_block(path: Path) -> gemmi.ReflnBlock | None:
    """Return the first reflection block of a validation map file, or None."""
    blocks = gemmi.as_refln_blocks(gemmi.cif.read(str(path)))
    return blocks[0] if blocks else None


def compute_validation_density(
    atoms: pd.DataFrame,
    two_fofc_path: Path | None,
    fofc_path: Path | None,
    *,
    deposited_cell: Cell | None = None,
) -> DensityComputation:
    """Return the wwPDB validation density columns of one structure.

    ``atoms`` is the structure's atom table, and the two paths are the wwPDB
    validation map files of the entry, or None when the archive has none.
    The maps were calculated about the deposited model, so they are read at
    the atoms' own positions and need no pairing with another model. The
    cell check is skipped without ``deposited_cell``.

    An entry without maps gets empty values and the reason as both statuses,
    with a warning. Raises what reading a file raises.
    """
    index = atoms.index
    if two_fofc_path is None or fofc_path is None:
        return _without_validation_maps(
            index,
            MAP_STATUS_NO_VALIDATION_MAPS,
            "The wwPDB archive has no validation map coefficients for this "
            "entry, so its wwPDB density columns are empty.",
        )
    blocks: list[gemmi.ReflnBlock] = []
    for path, labels in (
        (two_fofc_path, WWPDB_2FOFC_LABELS),
        (fofc_path, WWPDB_FOFC_LABELS),
    ):
        block = _read_reflection_block(path)
        present: set[str] = set(block.column_labels()) if block is not None else set()
        missing = [label for label in labels if label not in present]
        if block is None or missing:
            return _without_validation_maps(
                index,
                MAP_STATUS_NO_COEFFICIENTS,
                f"The wwPDB validation map file {path.name} lacks "
                + (", ".join(missing) if block is not None else "a reflection block")
                + ", so the wwPDB density columns are empty.",
            )
        blocks.append(block)
    for block in blocks:
        cell = block.cell
        map_cell: Cell = (cell.a, cell.b, cell.c, cell.alpha, cell.beta, cell.gamma)
        if deposited_cell is not None and not cells_agree(map_cell, deposited_cell):
            return _without_validation_maps(
                index,
                MAP_STATUS_CELL_MISMATCH,
                "The unit cell of the wwPDB validation maps "
                f"({', '.join(f'{value:g}' for value in map_cell)}) is not the "
                "deposited cell "
                f"({', '.join(f'{value:g}' for value in deposited_cell)}), so the "
                "wwPDB density columns are empty.",
            )
    rates = [choose_sample_rate(block) for block in blocks]
    if any(rate is None for rate in rates):
        return _without_validation_maps(
            index,
            MAP_STATUS_TOO_LARGE,
            f"The maps would exceed {MAP_MAX_GRID_POINTS:,} grid points at a "
            f"sample rate of {MAP_MIN_SAMPLE_RATE:g}, so the wwPDB density "
            "columns are empty.",
        )
    sample_rate = min(rate for rate in rates if rate is not None)
    positions = _positions(atoms, _DEPOSITED_POSITION_COLUMNS)
    located = np.isfinite(positions).all(axis=1)

    columns = _empty_validation_columns(index, MAP_STATUS_OK)
    rms_by_map: list[float] = []
    for block, labels, value_name, rms_name in zip(
        blocks,
        (WWPDB_2FOFC_LABELS, WWPDB_FOFC_LABELS),
        WWPDB_VALUE_COLUMNS,
        WWPDB_RMS_COLUMNS,
        strict=True,
    ):
        read = _read_map(block, labels, sample_rate, (positions,))
        if read is None:
            return _without_validation_maps(
                index,
                MAP_STATUS_NO_COEFFICIENTS,
                f"The {labels[0]} map of the wwPDB validation files has no "
                "usable density, so the wwPDB density columns are empty.",
            )
        rms, (values,) = read
        columns[value_name] = values / rms
        columns[rms_name] = rms
        rms_by_map.append(rms)
    # The difference map was the second.
    gradients = _read_gradients(blocks[1], WWPDB_FOFC_LABELS, sample_rate, positions)
    columns[list(WWPDB_GRADIENT_COLUMNS)] = gradients / rms_by_map[1]
    columns[WWPDB_MAP_RESOLUTION_COLUMN] = float(
        min(float(np.min(block.make_d_array())) for block in blocks)
    )
    columns[WWPDB_MAP_SAMPLE_RATE_COLUMN] = sample_rate
    status = np.full(len(index), ATOM_STATUS_OK, dtype=object)
    status[~located] = ATOM_STATUS_NO_COORDINATES
    columns[WWPDB_ATOM_STATUS_COLUMN] = status
    warnings: tuple[str, ...] = ()
    if sample_rate < MAP_SAMPLE_RATE:
        warnings = (
            f"The wwPDB validation maps were sampled at a rate of "
            f"{sample_rate:g} in place of {MAP_SAMPLE_RATE:g} to stay within "
            f"{MAP_MAX_GRID_POINTS:,} grid points.",
        )
    return DensityComputation(
        columns=columns,
        map_status=MAP_STATUS_OK,
        sample_rate=sample_rate,
        warnings=warnings,
    )


__all__ = [
    "Cell",
    "DensityComputation",
    "Reflections",
    "cells_agree",
    "choose_sample_rate",
    "compute_structure_density",
    "compute_validation_density",
    "level_shifts",
]
