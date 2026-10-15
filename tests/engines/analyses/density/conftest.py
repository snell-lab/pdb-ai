"""A synthetic reflection file for the density engine's tests.

The file is built with gemmi from a model of four carbon atoms in a 20 Å
cubic P 1 cell, at 2 Å resolution. Its FWT and PHWT columns hold the model's
own structure factors, so the 2mFo−DFc map has a peak at each atom. Its
DELFWT and PHDELWT columns hold the structure factors of the first atom
alone with the opposite sign, so the difference map has a hole at that atom
and is near zero at the others.

Its FC_ALL and PHIC_ALL columns hold FWT less DELFWT, as PDB-Redo's files
do. That is the model with its first atom at twice the weight, so the
calculated map has a peak twice as high there, and half of it is unsupported
by the data.

write_mtz can also write the data with the third atom moved, while the
calculated structure factors keep it where the model has it, so the
difference map has a slope at that atom and the shift that levels it is
known.

write_validation_maps writes the first two sets of coefficients as the two
reflection files the wwPDB validation pipeline produces, in mmCIF form.
Those files hold no calculated structure factors.
"""

from __future__ import annotations

from pathlib import Path

import gemmi
import numpy as np
import pandas as pd
import pytest

CELL = (20.0, 20.0, 20.0, 90.0, 90.0, 90.0)
RESOLUTION_A = 2.0
ATOM_POSITIONS = (
    (5.0, 5.0, 5.0),
    (10.0, 6.0, 5.0),
    (5.0, 12.0, 9.0),
    (14.0, 14.0, 15.0),
)


def _structure(positions: tuple[tuple[float, float, float], ...]) -> gemmi.Structure:
    structure = gemmi.Structure()
    structure.cell = gemmi.UnitCell(*CELL)
    structure.spacegroup_hm = "P 1"
    model = gemmi.Model(1)
    chain = gemmi.Chain("A")
    for number, (x, y, z) in enumerate(positions, start=1):
        residue = gemmi.Residue()
        residue.name = "ALA"
        residue.seqid = gemmi.SeqId(number, " ")
        atom = gemmi.Atom()
        atom.name = "CA"
        atom.element = gemmi.Element("C")
        atom.pos = gemmi.Position(x, y, z)
        atom.occ = 1.0
        atom.b_iso = 20.0
        residue.add_atom(atom)
        chain.add_residue(residue)
    model.add_chain(chain)
    structure.add_model(model)
    structure.setup_entities()
    return structure


def _structure_factors(structure: gemmi.Structure) -> gemmi.ComplexAsuData:
    calculator = gemmi.DensityCalculatorX()
    calculator.d_min = RESOLUTION_A
    calculator.set_grid_cell_and_spacegroup(structure)
    calculator.put_model_density_on_grid(structure[0])
    grid = gemmi.transform_map_to_f_phi(calculator.grid, half_l=True)
    return grid.prepare_asu_data(dmin=RESOLUTION_A)


MTZ_LABELS = ("FWT", "PHWT", "DELFWT", "PHDELWT", "FC_ALL", "PHIC_ALL")


def write_mtz(
    path: Path,
    *,
    labels: tuple[str, ...] = MTZ_LABELS,
    cell: tuple[float, float, float, float, float, float] = CELL,
    difference_scale: float = 1.0,
    displacement: tuple[float, float, float] | None = None,
) -> Path:
    """Write the synthetic reflection file to ``path`` and return the path.

    ``labels`` names the columns after h, k, and l, in the order of
    MTZ_LABELS, and a shorter tuple leaves the last columns out.
    ``difference_scale`` multiplies the difference amplitudes alone, so 0.5
    gives the file of a twinned refinement, whose difference coefficients
    are half of FWT less FC_ALL. ``displacement`` moves the third atom by
    that many Å in the data alone: FWT holds it moved, FC_ALL where the
    model has it, and DELFWT is still FWT less FC_ALL.
    """
    rows = np.array(_reflection_rows(displacement), dtype=np.float32)
    rows[:, 5] *= difference_scale
    mtz = gemmi.Mtz(with_base=True)
    mtz.spacegroup = gemmi.SpaceGroup("P 1")
    mtz.set_cell_for_all(gemmi.UnitCell(*cell))
    mtz.add_dataset("synthetic")
    for label, column_type in zip(labels, ("F", "P") * 3, strict=False):
        mtz.add_column(label, column_type)
    mtz.set_data(rows[:, : 3 + len(labels)])
    mtz.write_to_file(str(path))
    return path


def _reflection_rows(
    displacement: tuple[float, float, float] | None = None,
) -> list[list[float]]:
    """Return h, k, l and the six coefficients of every reflection.

    ``displacement`` moves the third atom in the 2mFo−DFc coefficients alone.
    """
    model = _structure_factors(_structure(ATOM_POSITIONS))
    hole = _structure_factors(_structure(ATOM_POSITIONS[:1]))
    hole_by_hkl = {
        tuple(hkl): value
        for hkl, value in zip(
            hole.miller_array.tolist(), hole.value_array.tolist(), strict=True
        )
    }
    observed_by_hkl = {
        tuple(hkl): value
        for hkl, value in zip(
            model.miller_array.tolist(), model.value_array.tolist(), strict=True
        )
    }
    if displacement is not None:
        x, y, z = ATOM_POSITIONS[2]
        dx, dy, dz = displacement
        moved = (*ATOM_POSITIONS[:2], (x + dx, y + dy, z + dz), *ATOM_POSITIONS[3:])
        shifted = _structure_factors(_structure(moved))
        observed_by_hkl = {
            tuple(hkl): value
            for hkl, value in zip(
                shifted.miller_array.tolist(), shifted.value_array.tolist(), strict=True
            )
        }
    rows: list[list[float]] = []
    for hkl, model_value in zip(
        model.miller_array.tolist(), model.value_array.tolist(), strict=True
    ):
        calculated = model_value + hole_by_hkl[tuple(hkl)]
        value = observed_by_hkl[tuple(hkl)]
        difference = value - calculated
        rows.append(
            [
                *(float(index) for index in hkl),
                float(abs(value)),
                float(np.degrees(np.angle(value))),
                float(abs(difference)),
                float(np.degrees(np.angle(difference))),
                float(abs(calculated)),
                float(np.degrees(np.angle(calculated))),
            ]
        )
    return rows


def write_validation_maps(
    directory: Path,
    *,
    two_fofc_items: tuple[str, str] = ("pdbx_FWT", "pdbx_PHWT"),
    cell: tuple[float, float, float, float, float, float] = CELL,
    displacement: tuple[float, float, float] | None = None,
) -> tuple[Path, Path]:
    """Write the two wwPDB-style map files to ``directory`` and return their paths.

    ``displacement`` moves the third atom in the data, as in write_mtz.
    """
    rows = _reflection_rows(displacement)
    header = (
        "data_1den\n"
        f"_cell.length_a {cell[0]}\n_cell.length_b {cell[1]}\n_cell.length_c {cell[2]}\n"
        f"_cell.angle_alpha {cell[3]}\n_cell.angle_beta {cell[4]}\n"
        f"_cell.angle_gamma {cell[5]}\n"
        "_symmetry.space_group_name_H-M 'P 1'\n"
        "loop_\n_refln.index_h\n_refln.index_k\n_refln.index_l\n"
    )
    paths: list[Path] = []
    for name, items, columns in (
        ("1den_2fofc_map_coef.cif", two_fofc_items, (3, 4)),
        ("1den_fofc_map_coef.cif", ("pdbx_DELFWT", "pdbx_DELPHWT"), (5, 6)),
    ):
        lines = [
            f"{int(row[0])} {int(row[1])} {int(row[2])} "
            f"{row[columns[0]]:.4f} {row[columns[1]]:.3f} 1.0"
            for row in rows
        ]
        path = directory / name
        path.write_text(
            header
            + f"_refln.{items[0]}\n_refln.{items[1]}\n_refln.fom\n"
            + "\n".join(lines)
            + "\n",
            encoding="utf-8",
        )
        paths.append(path)
    return paths[0], paths[1]


@pytest.fixture
def density_mtz(tmp_path: Path) -> Path:
    """Write the synthetic reflection file and return its path."""
    return write_mtz(tmp_path / "1den_final.mtz")


@pytest.fixture
def density_atoms() -> pd.DataFrame:
    """Return an atom table of the four atoms, each paired with PDB-Redo's model.

    PDB-Redo's position of the second atom is 0.3 Å along x from its
    deposited one. The index is not a plain count from zero, so a result
    that ignores it shows.
    """
    positions = np.array(ATOM_POSITIONS)
    pdb_redo = positions.copy()
    pdb_redo[1, 0] += 0.3
    return pd.DataFrame(
        {
            "x": positions[:, 0],
            "y": positions[:, 1],
            "z": positions[:, 2],
            "pdb_redo_x": pdb_redo[:, 0],
            "pdb_redo_y": pdb_redo[:, 1],
            "pdb_redo_z": pdb_redo[:, 2],
        },
        index=[10, 11, 12, 13],
    )
