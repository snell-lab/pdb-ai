from __future__ import annotations

from pathlib import Path

import gemmi
import numpy as np
import numpy.typing as npt
import pandas as pd
import pytest

from pdbai.engines.analyses.density import DENSITY_ENGINE
from pdbai.engines.analyses.density import compute as compute_module
from pdbai.engines.analyses.density.cell import read_cell_parameters
from pdbai.engines.analyses.density.columns import (
    DENSITY_COLUMNS,
    PDB_REDO_COLUMN_NAMES,
    PDB_REDO_GRADIENT_COLUMNS,
    PDB_REDO_SHIFT_COLUMNS,
    WWPDB_COLUMN_NAMES,
    WWPDB_GRADIENT_COLUMNS,
)
from pdbai.engines.analyses.density.compute import (
    cells_agree,
    choose_sample_rate,
    compute_structure_density,
    compute_validation_density,
    level_shifts,
)
from pdbai.engines.analyses.density.constants import (
    MAP_MIN_SAMPLE_RATE,
    MAP_SAMPLE_RATE,
)
from pdbai.engines.catalog import get_engine_registry
from pdbai.engines.compatibility import require_compatible_options
from pdbai.engines.core import ColumnGranularity, EngineContext, EngineOptions
from pdbai.engines.sources.pdb_redo import PDB_REDO_MTZ_PATH
from pdbai.engines.sources.validation.maps import (
    VALIDATION_2FOFC_MAP_PATH,
    VALIDATION_FOFC_MAP_PATH,
)
from pdbai.profiles import get_profile

from .conftest import (
    ATOM_POSITIONS,
    CELL,
    MTZ_LABELS,
    write_mtz,
    write_validation_maps,
)

_ON: dict[str, bool | str] = {
    "include_density": True,
    "density_map_source": "pdb_redo",
}
_WWPDB: dict[str, bool | str] = {
    "include_density": True,
    "density_map_source": "wwpdb",
}
_BOTH: dict[str, bool | str] = {
    "include_density": True,
    "density_map_source": "both",
}


def _values(frame: pd.DataFrame, column: str) -> list[float]:
    """Return a numeric column as plain floats in row order."""
    values: list[float] = frame[column].to_numpy(dtype=np.float64).tolist()
    return values


_SIGMA_COLUMNS = [
    "density_2fofc_sigma",
    "density_fofc_sigma",
    "density_2fofc_sigma_pdb_redo",
    "density_fofc_sigma_pdb_redo",
]
# The difference density and the calculated density on one scale.
_ABSOLUTE_COLUMNS = [
    "density_fofc_eA3",
    "density_fc_eA3",
    "density_fofc_eA3_pdb_redo",
    "density_fc_eA3_pdb_redo",
]
_VALUE_COLUMNS = [*_SIGMA_COLUMNS, *_ABSOLUTE_COLUMNS]


def test_each_atom_sits_on_a_peak_of_the_2fofc_map(
    density_mtz: Path, density_atoms: pd.DataFrame
) -> None:
    computation = compute_structure_density(density_atoms, density_mtz)
    columns = computation.columns

    assert computation.map_status == "ok"
    assert computation.sample_rate == MAP_SAMPLE_RATE
    assert columns.index.equals(density_atoms.index)
    assert (columns["density_2fofc_sigma"] > 10).all()
    assert (columns["density_status"] == "ok").all()
    assert (columns["density_map_status"] == "ok").all()
    assert (columns["density_map_sample_rate"] == MAP_SAMPLE_RATE).all()
    assert _values(columns, "density_map_resolution_A")[0] == pytest.approx(
        2.0, abs=0.05
    )
    assert columns["density_2fofc_rms"].nunique() == 1
    assert _values(columns, "density_2fofc_rms")[0] > 0


def test_the_difference_map_has_its_hole_at_the_first_atom_alone(
    density_mtz: Path, density_atoms: pd.DataFrame
) -> None:
    columns = compute_structure_density(density_atoms, density_mtz).columns

    assert _values(columns, "density_fofc_sigma")[0] < -20
    assert max(abs(value) for value in _values(columns, "density_fofc_sigma")[1:]) < 2
    assert _values(columns, "density_fofc_sigma_pdb_redo")[0] == pytest.approx(
        _values(columns, "density_fofc_sigma")[0]
    )


def test_the_two_positions_of_an_atom_are_read_apart(
    density_mtz: Path, density_atoms: pd.DataFrame
) -> None:
    columns = compute_structure_density(density_atoms, density_mtz).columns

    # PDB-Redo's second atom lies 0.3 Å off the peak the deposited one is on.
    assert (
        _values(columns, "density_2fofc_sigma_pdb_redo")[1]
        < _values(columns, "density_2fofc_sigma")[1] - 0.5
    )
    for row in (0, 2, 3):
        assert _values(columns, "density_2fofc_sigma_pdb_redo")[row] == pytest.approx(
            _values(columns, "density_2fofc_sigma")[row]
        )


def test_a_value_is_the_map_in_units_of_its_root_mean_square(
    density_mtz: Path, density_atoms: pd.DataFrame
) -> None:
    columns = compute_structure_density(density_atoms, density_mtz).columns
    # The same map on a much finer grid, normalized by gemmi itself.
    grid = gemmi.read_mtz_file(str(density_mtz)).transform_f_phi_to_map(
        "FWT", "PHWT", sample_rate=10.0
    )
    rms = float(np.asarray(grid).std(dtype=np.float64))
    grid.normalize()
    fine = [
        grid.tricubic_interpolation(gemmi.Position(*position))
        for position in ATOM_POSITIONS
    ]

    assert columns["density_2fofc_sigma"].tolist() == pytest.approx(fine, abs=0.15)
    assert _values(columns, "density_2fofc_rms")[0] == pytest.approx(rms, rel=1e-3)


def test_the_calculated_density_is_the_map_on_the_scale_of_its_coefficients(
    density_mtz: Path, density_atoms: pd.DataFrame
) -> None:
    columns = compute_structure_density(density_atoms, density_mtz).columns
    # The same map on a much finer grid, left on the scale gemmi gives it.
    grid = gemmi.read_mtz_file(str(density_mtz)).transform_f_phi_to_map(
        "FC_ALL", "PHIC_ALL", sample_rate=10.0
    )
    fine = [
        grid.tricubic_interpolation(gemmi.Position(*position))
        for position in ATOM_POSITIONS
    ]

    calculated = _values(columns, "density_fc_eA3")
    assert calculated == pytest.approx(fine, abs=0.01)
    # A carbon atom of B 20 Å² at 2 Å is a peak of about 1 e/Å³, and the
    # model holds the first atom twice.
    assert calculated[1:] == pytest.approx([1.04, 1.04, 1.04], abs=0.02)
    assert calculated[0] == pytest.approx(2 * calculated[2], rel=0.01)
    # PDB-Redo's second atom lies 0.3 Å off the peak.
    at_pdb_redo = _values(columns, "density_fc_eA3_pdb_redo")
    assert at_pdb_redo[1] < calculated[1] - 0.05
    for row in (0, 2, 3):
        assert at_pdb_redo[row] == pytest.approx(calculated[row])


def test_the_difference_over_the_calculated_density_is_a_fraction(
    density_mtz: Path, density_atoms: pd.DataFrame
) -> None:
    columns = compute_structure_density(density_atoms, density_mtz).columns

    # The data support half of what the model places at the first atom, and
    # all of what it places at the others.
    for suffix in ("", "_pdb_redo"):
        ratio = (
            columns[f"density_fofc_eA3{suffix}"] / columns[f"density_fc_eA3{suffix}"]
        ).tolist()
        assert ratio[0] == pytest.approx(-0.5, abs=0.005)
        assert ratio[1:] == pytest.approx([0.0, 0.0, 0.0], abs=0.01)
    # The two sigma values of the first atom have no such reading.
    sigma_ratio = (columns["density_fofc_sigma"] / columns["density_2fofc_sigma"])[10]
    assert sigma_ratio == pytest.approx(-2.0, abs=0.01)


def test_the_difference_density_is_on_one_scale_in_a_twinned_refinement(
    tmp_path: Path, density_mtz: Path, density_atoms: pd.DataFrame
) -> None:
    columns = compute_structure_density(density_atoms, density_mtz).columns
    # A twinned refinement writes half the difference coefficients.
    twinned = compute_structure_density(
        density_atoms, write_mtz(tmp_path / "twinned.mtz", difference_scale=0.5)
    ).columns

    for suffix in ("", "_pdb_redo"):
        difference = columns[f"density_fofc_eA3{suffix}"]
        own_map = columns[f"density_fofc_sigma{suffix}"] * columns["density_fofc_rms"]
        assert difference.tolist() == pytest.approx(own_map.tolist(), abs=1e-5)
        twinned_map = (
            twinned[f"density_fofc_sigma{suffix}"] * twinned["density_fofc_rms"]
        )
        assert difference.tolist() == pytest.approx(
            (2 * twinned_map).tolist(), abs=1e-5
        )
    pd.testing.assert_frame_equal(
        twinned[_ABSOLUTE_COLUMNS], columns[_ABSOLUTE_COLUMNS]
    )


def test_a_file_without_calculated_structure_factors_keeps_the_two_maps(
    tmp_path: Path, density_mtz: Path, density_atoms: pd.DataFrame
) -> None:
    expected = compute_structure_density(density_atoms, density_mtz).columns
    lacking = write_mtz(tmp_path / "lacking.mtz", labels=MTZ_LABELS[:4])
    # A file whose calculated structure factors are all zero gives a flat map.
    flat = gemmi.read_mtz_file(str(density_mtz))
    data = np.array(flat, copy=True)
    data[:, [column.idx for column in flat.columns if column.label == "FC_ALL"]] = 0.0
    flat.set_data(data)
    flat.write_to_file(str(tmp_path / "flat.mtz"))

    for path, reason in (
        (lacking, "lacks the calculated structure factor column(s) FC_ALL, PHIC_ALL"),
        (tmp_path / "flat.mtz", "gives a FC_ALL map without usable density"),
    ):
        computation = compute_structure_density(density_atoms, path)
        columns = computation.columns

        assert computation.map_status == "ok"
        assert list(columns.columns) == list(PDB_REDO_COLUMN_NAMES)
        assert columns[_ABSOLUTE_COLUMNS].isna().all().all()
        pd.testing.assert_frame_equal(columns[_SIGMA_COLUMNS], expected[_SIGMA_COLUMNS])
        assert (columns["density_status"] == "ok").all()
        assert len(computation.warnings) == 1
        assert reason in computation.warnings[0]


# The data move the third atom by about 0.1 Å, in no axis's direction.
_DISPLACEMENT = (0.08, -0.05, 0.04)


def _gradients(frame: pd.DataFrame, names: tuple[str, ...]) -> npt.NDArray[np.float64]:
    """Return three columns as an array with a row per atom."""
    return frame[list(names)].to_numpy(dtype=np.float64)


def test_the_gradient_is_the_slope_of_the_difference_map_in_sigma(
    tmp_path: Path, density_atoms: pd.DataFrame
) -> None:
    # An oblique cell checks the turn from fractional to orthogonal axes.
    path = write_mtz(
        tmp_path / "oblique.mtz",
        cell=(20.0, 22.0, 19.0, 80.0, 100.0, 110.0),
        displacement=_DISPLACEMENT,
    )
    columns = compute_structure_density(density_atoms, path).columns
    # Central differences of the same map on a much finer grid.
    grid = gemmi.read_mtz_file(str(path)).transform_f_phi_to_map(
        "DELFWT", "PHDELWT", sample_rate=12.0
    )
    rms = _values(columns, "density_fofc_rms")[0]
    positions = density_atoms[["pdb_redo_x", "pdb_redo_y", "pdb_redo_z"]].to_numpy()
    step = 0.005

    slopes = np.empty((len(positions), 3))
    for axis in range(3):
        offset = np.zeros(3)
        offset[axis] = step
        ahead, behind = (
            np.array(
                [
                    grid.tricubic_interpolation(gemmi.Position(*point))
                    for point in points
                ]
            )
            for points in (positions + offset, positions - offset)
        )
        slopes[:, axis] = (ahead - behind) / (2 * step) / rms

    gradients = _gradients(columns, PDB_REDO_GRADIENT_COLUMNS)
    assert np.abs(slopes).max() > 1.0
    assert gradients == pytest.approx(slopes, abs=0.02 * np.abs(slopes).max())


def test_the_gradient_points_where_the_data_put_the_atom(
    tmp_path: Path, density_mtz: Path, density_atoms: pd.DataFrame
) -> None:
    level = compute_structure_density(density_atoms, density_mtz).columns
    moved = compute_structure_density(
        density_atoms, write_mtz(tmp_path / "moved.mtz", displacement=_DISPLACEMENT)
    ).columns

    # A symmetric hole has no slope at its centre, and the third atom has
    # none until the data move it.
    level_gradients = _gradients(level, PDB_REDO_GRADIENT_COLUMNS)
    assert np.abs(level_gradients[0]).max() < 0.01
    gradient = _gradients(moved, PDB_REDO_GRADIENT_COLUMNS)[2]
    assert np.linalg.norm(gradient) > 10 * np.linalg.norm(level_gradients[2])
    direction = np.array(_DISPLACEMENT) / np.linalg.norm(_DISPLACEMENT)
    assert gradient @ direction > 0.99 * np.linalg.norm(gradient)
    assert np.linalg.norm(gradient) > 3.0
    # The deposited position has no gradient column.
    assert all(
        name.endswith("_pdb_redo")
        for name in PDB_REDO_COLUMN_NAMES
        if "gradient" in name or "shift" in name
    )


def test_the_shift_recovers_how_far_the_data_moved_the_atom(
    tmp_path: Path, density_atoms: pd.DataFrame
) -> None:
    columns = compute_structure_density(
        density_atoms, write_mtz(tmp_path / "moved.mtz", displacement=_DISPLACEMENT)
    ).columns
    shifts = _gradients(columns, PDB_REDO_SHIFT_COLUMNS)

    assert shifts[2] == pytest.approx(_DISPLACEMENT, abs=0.01)
    # The others are still, but for the ripples of the hole at the first.
    for row in (0, 1, 3):
        assert np.abs(shifts[row]).max() < 0.02
    # A twinned refinement halves the difference coefficients, and neither
    # the gradient in sigma nor the shift changes.
    twinned = compute_structure_density(
        density_atoms,
        write_mtz(
            tmp_path / "twinned.mtz", displacement=_DISPLACEMENT, difference_scale=0.5
        ),
    ).columns
    for names in (PDB_REDO_GRADIENT_COLUMNS, PDB_REDO_SHIFT_COLUMNS):
        assert _gradients(twinned, names) == pytest.approx(
            _gradients(columns, names), abs=1e-4
        )


def test_without_calculated_structure_factors_there_is_a_gradient_and_no_shift(
    tmp_path: Path, density_atoms: pd.DataFrame
) -> None:
    full = compute_structure_density(
        density_atoms, write_mtz(tmp_path / "full.mtz", displacement=_DISPLACEMENT)
    )
    lacking = compute_structure_density(
        density_atoms,
        write_mtz(
            tmp_path / "lacking.mtz", labels=MTZ_LABELS[:4], displacement=_DISPLACEMENT
        ),
    )

    pd.testing.assert_frame_equal(
        lacking.columns[list(PDB_REDO_GRADIENT_COLUMNS)],
        full.columns[list(PDB_REDO_GRADIENT_COLUMNS)],
    )
    assert lacking.columns[list(PDB_REDO_SHIFT_COLUMNS)].isna().all().all()
    assert "and the shifts are empty" in lacking.warnings[0]


def test_an_atom_without_a_partner_has_no_gradient(
    density_mtz: Path, density_atoms: pd.DataFrame
) -> None:
    atoms = density_atoms.copy()
    atoms.loc[12, ["pdb_redo_x", "pdb_redo_y", "pdb_redo_z"]] = np.nan

    columns = compute_structure_density(atoms, density_mtz).columns

    for names in (PDB_REDO_GRADIENT_COLUMNS, PDB_REDO_SHIFT_COLUMNS):
        missing = columns[list(names)].isna().all(axis=1)
        assert missing.tolist() == [False, False, True, False]


def test_a_shift_needs_a_curvature_that_falls_in_every_direction() -> None:
    slopes = np.array([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0], [np.nan, 0.0, 0.0]])
    curvatures = np.array(
        [np.diag([-2.0, -4.0, -4.0]), np.diag([-2.0, -4.0, 0.5]), np.diag([-2.0] * 3)]
    )

    shifts = level_shifts(slopes, curvatures)

    assert shifts[0].tolist() == pytest.approx([0.5, 0.0, 0.0])
    assert np.isnan(shifts[1:]).all()


def test_the_validation_gradient_is_read_at_the_deposited_position(
    tmp_path: Path, density_atoms: pd.DataFrame
) -> None:
    two_fofc, fofc = write_validation_maps(tmp_path, displacement=_DISPLACEMENT)
    pdb_redo = compute_structure_density(
        density_atoms,
        write_mtz(tmp_path / "1den_final.mtz", displacement=_DISPLACEMENT),
    ).columns
    wwpdb = compute_validation_density(density_atoms, two_fofc, fofc).columns

    # The files hold the same coefficients, rounded when written as text, and
    # the two positions are one point for every atom but the second.
    assert _gradients(wwpdb, WWPDB_GRADIENT_COLUMNS)[[0, 2, 3]] == pytest.approx(
        _gradients(pdb_redo, PDB_REDO_GRADIENT_COLUMNS)[[0, 2, 3]], abs=0.05
    )
    assert np.linalg.norm(_gradients(wwpdb, WWPDB_GRADIENT_COLUMNS)[2]) > 3.0
    unlocated = density_atoms.copy()
    unlocated.loc[13, "x"] = np.nan
    gradients = compute_validation_density(unlocated, two_fofc, fofc).columns
    assert gradients[list(WWPDB_GRADIENT_COLUMNS)].isna().all(axis=1).tolist() == [
        False,
        False,
        False,
        True,
    ]


def test_an_entry_without_a_reflection_file_has_empty_values(
    density_atoms: pd.DataFrame,
) -> None:
    computation = compute_structure_density(density_atoms, None)

    assert computation.map_status == "no_pdb_redo_entry"
    assert computation.sample_rate is None
    assert computation.columns[_VALUE_COLUMNS].isna().all().all()
    assert (computation.columns["density_status"] == "no_pdb_redo_entry").all()
    assert computation.warnings and "no reflection file" in computation.warnings[0]


def test_a_file_without_the_map_coefficients_gives_no_values(
    tmp_path: Path, density_atoms: pd.DataFrame
) -> None:
    path = write_mtz(tmp_path / "other.mtz", labels=("FP", "PHIC", "DELFWT", "PHDELWT"))

    computation = compute_structure_density(density_atoms, path)

    assert computation.map_status == "no_map_coefficients"
    assert "FWT, PHWT" in computation.warnings[0]
    assert computation.columns[_VALUE_COLUMNS].isna().all().all()


def test_a_reflection_file_of_another_cell_gives_no_values(
    density_mtz: Path, density_atoms: pd.DataFrame
) -> None:
    computation = compute_structure_density(
        density_atoms, density_mtz, deposited_cell=(20.0, 20.0, 40.0, 90.0, 90.0, 90.0)
    )

    assert computation.map_status == "cell_mismatch"
    assert computation.columns[_VALUE_COLUMNS].isna().all().all()
    assert (
        compute_structure_density(
            density_atoms, density_mtz, deposited_cell=CELL
        ).map_status
        == "ok"
    )


def test_cells_agree_within_a_hundredth_and_a_degree() -> None:
    assert cells_agree(CELL, (20.1, 20.0, 19.9, 90.5, 90.0, 89.5))
    assert not cells_agree(CELL, (20.5, 20.0, 20.0, 90.0, 90.0, 90.0))
    assert not cells_agree(CELL, (20.0, 20.0, 20.0, 90.0, 92.0, 90.0))


def test_models_that_never_paired_give_no_values(
    density_mtz: Path, density_atoms: pd.DataFrame
) -> None:
    atoms = density_atoms.drop(columns=["pdb_redo_x", "pdb_redo_y", "pdb_redo_z"])

    computation = compute_structure_density(atoms, density_mtz)

    assert computation.map_status == "models_not_paired"
    assert computation.columns[_VALUE_COLUMNS].isna().all().all()


def test_an_unpaired_atom_keeps_its_deposited_values(
    density_mtz: Path, density_atoms: pd.DataFrame
) -> None:
    atoms = density_atoms.copy()
    atoms.loc[12, ["pdb_redo_x", "pdb_redo_y", "pdb_redo_z"]] = np.nan
    atoms.loc[13, "x"] = np.nan

    columns = compute_structure_density(atoms, density_mtz).columns

    assert columns["density_status"].tolist() == [
        "ok",
        "ok",
        "no_pdb_redo_atom",
        "no_coordinates",
    ]
    deposited = _values(columns, "density_2fofc_sigma")
    at_pdb_redo = _values(columns, "density_2fofc_sigma_pdb_redo")
    assert deposited[2] > 10
    assert np.isnan(at_pdb_redo[2])
    assert np.isnan(deposited[3])
    assert at_pdb_redo[3] > 10
    # The calculated and the difference density are missing where the sigma
    # values are.
    assert columns[_ABSOLUTE_COLUMNS].isna().to_numpy().tolist() == [
        [False, False, False, False],
        [False, False, False, False],
        [False, False, True, True],
        [True, True, False, False],
    ]


def test_a_large_cell_is_sampled_more_coarsely_or_not_at_all(
    density_mtz: Path, density_atoms: pd.DataFrame, monkeypatch: pytest.MonkeyPatch
) -> None:
    mtz = gemmi.read_mtz_file(str(density_mtz))
    at_full_rate = int(np.prod(mtz.get_size_for_hkl(sample_rate=MAP_SAMPLE_RATE)))
    at_lowest_rate = int(np.prod(mtz.get_size_for_hkl(sample_rate=MAP_MIN_SAMPLE_RATE)))

    monkeypatch.setattr(compute_module, "MAP_MAX_GRID_POINTS", at_full_rate - 1)
    lowered = choose_sample_rate(mtz)
    assert lowered is not None
    assert MAP_MIN_SAMPLE_RATE <= lowered < MAP_SAMPLE_RATE
    computation = compute_structure_density(density_atoms, density_mtz)
    assert computation.map_status == "ok"
    assert computation.sample_rate == lowered
    assert computation.warnings and "sampled at a rate" in computation.warnings[0]

    monkeypatch.setattr(compute_module, "MAP_MAX_GRID_POINTS", at_lowest_rate - 1)
    assert choose_sample_rate(mtz) is None
    assert compute_structure_density(density_atoms, density_mtz).map_status == (
        "map_too_large"
    )


def test_compute_returns_exactly_the_declared_columns(
    density_mtz: Path, density_atoms: pd.DataFrame
) -> None:
    declared = list(PDB_REDO_COLUMN_NAMES)

    assert list(compute_structure_density(density_atoms, density_mtz).columns) == (
        declared
    )
    assert list(compute_structure_density(density_atoms, None).columns) == declared
    assert [column.name for column in DENSITY_COLUMNS] == [
        *PDB_REDO_COLUMN_NAMES,
        *WWPDB_COLUMN_NAMES,
    ]
    structure_level = {
        column.name
        for column in DENSITY_COLUMNS
        if column.granularity is ColumnGranularity.STRUCTURE
    }
    names = {
        "density_2fofc_rms",
        "density_fofc_rms",
        "density_map_resolution_A",
        "density_map_sample_rate",
        "density_map_status",
    }
    assert structure_level == names | {f"{name}_wwpdb" for name in names}


def test_the_deposited_cell_is_read_from_the_atom_table() -> None:
    atoms = pd.DataFrame(
        {
            "_cell.length_a": ["20.000", "20.000"],
            "_cell.length_b": ["20.000", "20.000"],
            "_cell.length_c": ["20.000", "20.000"],
            "_cell.angle_alpha": ["90.00", "90.00"],
            "_cell.angle_beta": ["90.00", "90.00"],
            "_cell.angle_gamma": ["90.00", "90.00"],
        }
    )

    assert read_cell_parameters(atoms) == CELL
    assert read_cell_parameters(atoms.drop(columns=["_cell.angle_beta"])) is None
    assert read_cell_parameters(atoms.assign(**{"_cell.length_a": "?"})) is None
    assert read_cell_parameters(atoms.iloc[:0]) is None
    # A standard uncertainty is dropped, as the parser and every other engine
    # that reads the cell drop it.
    assert read_cell_parameters(atoms.assign(**{"_cell.length_a": "20.000(4)"})) == (
        CELL
    )


def _context(atoms: pd.DataFrame, path: Path | None) -> EngineContext:
    context = EngineContext("1DEN", atoms, EngineOptions(values=_ON))
    if path is not None:
        context.artifacts.put(PDB_REDO_MTZ_PATH, path)
    return context


def test_the_engine_reports_the_maps(
    density_mtz: Path, density_atoms: pd.DataFrame
) -> None:
    result = DENSITY_ENGINE.run(_context(density_atoms, density_mtz))

    assert result.columns is not None
    assert list(result.columns.columns) == list(PDB_REDO_COLUMN_NAMES)
    assert result.report == {
        "density_map_status": "ok",
        "density_map_sample_rate": MAP_SAMPLE_RATE,
        "density_message": None,
    }
    assert result.warnings == ()


def test_the_engine_warns_once_about_an_entry_pdb_redo_lacks(
    density_atoms: pd.DataFrame,
) -> None:
    result = DENSITY_ENGINE.run(_context(density_atoms, None))

    assert result.report["density_map_status"] == "no_pdb_redo_entry"
    assert len(result.warnings) == 1
    assert result.warnings[0].startswith("Density: 1DEN: ")

    failed = _context(density_atoms, None)
    failed.failures["pdb_redo_map_source"] = "download failed"
    assert DENSITY_ENGINE.run(failed).warnings == ()


def test_the_engine_is_off_by_default_and_for_x_ray_downloads_only() -> None:
    registry = get_engine_registry()

    assert registry.with_defaults(EngineOptions()).flag("include_density") is False
    assert DENSITY_ENGINE.spec.techniques == frozenset({"x_ray_diffraction"})
    assert DENSITY_ENGINE.spec.sources == frozenset({"download"})
    with pytest.raises(ValueError, match="include_density"):
        require_compatible_options(
            registry.with_defaults(
                EngineOptions(technique="neutron_diffraction", values=_ON)
            )
        )


def test_the_engine_brings_the_pdb_redo_engines_it_reads_from() -> None:
    registry = get_engine_registry()
    options = registry.with_defaults(EngineOptions(values=_ON))

    names = [
        engine.spec.name
        for phase in ("acquisition", "annotation")
        for engine in registry.engines_to_run(
            phase=phase,  # type: ignore[arg-type]
            options=options,
            source="download",
        )
    ]

    assert {
        "pdb_redo_map_source",
        "pdb_redo_source",
        "pdb_redo_structure",
        "pdb_redo_comparison",
        "density",
    } <= set(names)
    assert names.index("pdb_redo_comparison") < names.index("density")


def test_the_validation_maps_are_read_at_the_deposited_position(
    tmp_path: Path, density_atoms: pd.DataFrame
) -> None:
    two_fofc, fofc = write_validation_maps(tmp_path)
    # These maps belong to the deposited model, so no pairing is needed.
    atoms = density_atoms[["x", "y", "z"]]

    computation = compute_validation_density(atoms, two_fofc, fofc, deposited_cell=CELL)
    columns = computation.columns

    assert computation.map_status == "ok"
    assert computation.sample_rate == MAP_SAMPLE_RATE
    assert list(columns.columns) == list(WWPDB_COLUMN_NAMES)
    assert columns.index.equals(atoms.index)
    assert min(_values(columns, "density_2fofc_sigma_wwpdb")) > 10
    holes = _values(columns, "density_fofc_sigma_wwpdb")
    assert holes[0] < -20
    assert max(abs(value) for value in holes[1:]) < 2
    assert (columns["density_status_wwpdb"] == "ok").all()
    assert _values(columns, "density_map_resolution_A_wwpdb")[0] == pytest.approx(
        2.0, abs=0.05
    )
    # The validation files hold no calculated structure factors.
    assert not [name for name in WWPDB_COLUMN_NAMES if "eA3" in name]


def test_both_sources_read_the_same_map_alike(
    tmp_path: Path, density_mtz: Path, density_atoms: pd.DataFrame
) -> None:
    two_fofc, fofc = write_validation_maps(tmp_path)

    pdb_redo = compute_structure_density(density_atoms, density_mtz).columns
    wwpdb = compute_validation_density(density_atoms, two_fofc, fofc).columns

    # The files hold the same coefficients, rounded when written as text.
    assert _values(wwpdb, "density_2fofc_sigma_wwpdb") == pytest.approx(
        _values(pdb_redo, "density_2fofc_sigma"), abs=0.05
    )
    assert _values(wwpdb, "density_fofc_sigma_wwpdb") == pytest.approx(
        _values(pdb_redo, "density_fofc_sigma"), abs=0.05
    )


def test_an_entry_without_validation_maps_has_empty_values(
    tmp_path: Path, density_atoms: pd.DataFrame
) -> None:
    two_fofc, fofc = write_validation_maps(tmp_path)

    for paths in ((None, None), (two_fofc, None), (None, fofc)):
        computation = compute_validation_density(density_atoms, *paths)
        assert computation.map_status == "no_validation_maps"
        assert computation.columns["density_fofc_sigma_wwpdb"].isna().all()
        assert (
            computation.columns["density_status_wwpdb"] == "no_validation_maps"
        ).all()


def test_validation_maps_are_checked_like_any_other(
    tmp_path: Path, density_atoms: pd.DataFrame
) -> None:
    two_fofc, fofc = write_validation_maps(tmp_path)
    renamed = tmp_path / "renamed"
    renamed.mkdir()
    other_items, _fofc = write_validation_maps(renamed, two_fofc_items=("FWT", "PHWT"))

    assert (
        compute_validation_density(
            density_atoms,
            two_fofc,
            fofc,
            deposited_cell=(20.0, 20.0, 40.0, 90.0, 90.0, 90.0),
        ).map_status
        == "cell_mismatch"
    )
    lacking = compute_validation_density(density_atoms, other_items, fofc)
    assert lacking.map_status == "no_map_coefficients"
    assert "pdbx_FWT, pdbx_PHWT" in lacking.warnings[0]
    unlocated = density_atoms.copy()
    unlocated.loc[13, "x"] = np.nan
    statuses = compute_validation_density(unlocated, two_fofc, fofc).columns[
        "density_status_wwpdb"
    ]
    assert statuses.tolist() == ["ok", "ok", "ok", "no_coordinates"]


def test_the_engine_reads_the_sources_the_run_names(
    tmp_path: Path, density_mtz: Path, density_atoms: pd.DataFrame
) -> None:
    two_fofc, fofc = write_validation_maps(tmp_path)

    def run(values: dict[str, bool | str]) -> tuple[list[str], dict[str, object]]:
        context = EngineContext("1DEN", density_atoms, EngineOptions(values=values))
        context.artifacts.put(PDB_REDO_MTZ_PATH, density_mtz)
        context.artifacts.put(VALIDATION_2FOFC_MAP_PATH, two_fofc)
        context.artifacts.put(VALIDATION_FOFC_MAP_PATH, fofc)
        result = DENSITY_ENGINE.run(context)
        assert result.columns is not None
        return list(result.columns.columns), dict(result.report)

    columns, report = run(_WWPDB)
    assert columns == list(WWPDB_COLUMN_NAMES)
    assert report == {
        "density_map_status_wwpdb": "ok",
        "density_map_sample_rate_wwpdb": MAP_SAMPLE_RATE,
        "density_message": None,
    }
    columns, report = run(_BOTH)
    assert columns == [*PDB_REDO_COLUMN_NAMES, *WWPDB_COLUMN_NAMES]
    assert report["density_map_status"] == "ok"
    assert report["density_map_status_wwpdb"] == "ok"


def test_each_source_brings_only_its_own_engines() -> None:
    registry = get_engine_registry()

    def names(values: dict[str, bool | str]) -> set[str]:
        options = registry.with_defaults(EngineOptions(values=values))
        return {
            engine.spec.name
            for phase in ("acquisition", "annotation")
            for engine in registry.engines_to_run(
                phase=phase,  # type: ignore[arg-type]
                options=options,
                source="download",
            )
        }

    wwpdb, both = names(_WWPDB), names(_BOTH)
    assert "validation_map_source" in wwpdb
    assert not {"pdb_redo_map_source", "pdb_redo_comparison"} & wwpdb
    assert {
        "validation_map_source",
        "pdb_redo_map_source",
        "pdb_redo_comparison",
    } <= both
    assert "validation_map_source" not in names({"include_density": True})
    # A run without the density engine fetches no maps at all.
    assert not {"validation_map_source", "pdb_redo_map_source"} & names(
        {"density_map_source": "both"}
    )


def test_a_run_writes_the_columns_of_the_sources_it_reads() -> None:
    registry = get_engine_registry()

    def enabled(values: dict[str, bool | str]) -> set[str]:
        options = registry.with_defaults(EngineOptions(values=values))
        return {
            column.name
            for column in DENSITY_COLUMNS
            if column.is_enabled(options, source="download")
        }

    assert enabled({"include_density": True}) == set(PDB_REDO_COLUMN_NAMES)
    assert enabled(_WWPDB) == set(WWPDB_COLUMN_NAMES)
    assert enabled(_BOTH) == {*PDB_REDO_COLUMN_NAMES, *WWPDB_COLUMN_NAMES}


def test_only_the_radiation_profile_turns_the_engine_on() -> None:
    assert "include_density" in get_profile("radiation_sensitivity").toggles
    assert "include_density" not in get_profile("research_ready").toggles
    assert "include_density" not in get_profile("ml_minimal").toggles
