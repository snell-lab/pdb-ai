from __future__ import annotations

import math

import pytest

from pdbai.structure.mmcif_fields import (
    calculate_unit_cell_volume,
    expand_mmcif_atom_fields,
)


def _triclinic_volume(
    a: float, b: float, c: float, alpha: float, beta: float, gamma: float
) -> float:
    cos_alpha, cos_beta, cos_gamma = (
        math.cos(math.radians(angle)) for angle in (alpha, beta, gamma)
    )
    return (
        a
        * b
        * c
        * math.sqrt(
            1
            - cos_alpha**2
            - cos_beta**2
            - cos_gamma**2
            + 2 * cos_alpha * cos_beta * cos_gamma
        )
    )


def test_orthorhombic_cell_volume_is_the_product_of_its_edges() -> None:
    assert calculate_unit_cell_volume(40, 50, 60, 90, 90, 90) == pytest.approx(
        120_000.0
    )


def test_hexagonal_cell_volume_uses_the_sine_of_gamma() -> None:
    assert calculate_unit_cell_volume(10, 10, 20, 90, 90, 120) == pytest.approx(
        10 * 10 * 20 * math.sqrt(3) / 2
    )


def test_rhombohedral_cell_of_sixty_degree_angles_has_the_known_volume() -> None:
    # This is the primitive cell of a face-centred cubic lattice, whose
    # volume is a³/√2.
    assert calculate_unit_cell_volume(10, 10, 10, 60, 60, 60) == pytest.approx(
        1000 / math.sqrt(2)
    )


def test_monoclinic_crambin_cell_volume() -> None:
    # This is the cell of PDB entry 1CRN, crambin, in space group P 1 21 1.
    assert calculate_unit_cell_volume(
        40.96, 18.65, 22.52, 90, 90.77, 90
    ) == pytest.approx(40.96 * 18.65 * 22.52 * math.sin(math.radians(90.77)))


@pytest.mark.parametrize(
    "cell",
    [
        (27.24, 31.87, 34.23, 88.52, 108.53, 111.89),
        (10, 20, 30, 100, 110, 120),
        (50, 50, 50, 30, 40, 50),
    ],
)
def test_triclinic_cell_volume_matches_the_standard_expression(
    cell: tuple[float, float, float, float, float, float],
) -> None:
    assert calculate_unit_cell_volume(*cell) == pytest.approx(_triclinic_volume(*cell))


@pytest.mark.parametrize("missing", range(6))
def test_cell_with_a_missing_parameter_has_no_volume(missing: int) -> None:
    parameters: list[float | None] = [40, 50, 60, 90, 90, 90]
    parameters[missing] = None

    assert math.isnan(calculate_unit_cell_volume(*parameters))


def test_absent_cell_has_no_volume() -> None:
    assert math.isnan(calculate_unit_cell_volume(None, None, None, None, None, None))


@pytest.mark.parametrize("value", [math.nan, math.inf, -math.inf])
@pytest.mark.parametrize("position", [0, 4])
def test_cell_with_a_non_finite_parameter_has_no_volume(
    value: float, position: int
) -> None:
    parameters: list[float | None] = [40, 50, 60, 90, 90, 90]
    parameters[position] = value

    assert math.isnan(calculate_unit_cell_volume(*parameters))


@pytest.mark.parametrize("length", [0.0, -40.0])
def test_cell_with_a_non_positive_length_has_no_volume(length: float) -> None:
    assert math.isnan(calculate_unit_cell_volume(40, length, 60, 90, 90, 90))


@pytest.mark.parametrize("angle", [0.0, -90.0, 180.0, 200.0, 270.0])
def test_cell_with_an_angle_outside_zero_to_180_degrees_has_no_volume(
    angle: float,
) -> None:
    # An angle of 200 degrees has the cosine of 160 degrees, so the volume
    # expression alone would give it a positive volume.
    assert math.isnan(calculate_unit_cell_volume(10, 20, 30, angle, 90, 90))


@pytest.mark.parametrize(
    "angles",
    [
        (120, 120, 120),
        (170, 170, 170),
        (90, 30, 120),
        (90, 30, 130),
        (30, 130, 90),
    ],
)
def test_angles_that_cannot_close_a_cell_give_no_volume(
    angles: tuple[float, float, float],
) -> None:
    assert math.isnan(calculate_unit_cell_volume(10, 20, 30, *angles))


def test_wwpdb_placeholder_cell_of_non_crystal_entries_has_no_volume() -> None:
    assert math.isnan(calculate_unit_cell_volume(1, 1, 1, 90, 90, 90))


def test_small_cell_that_is_not_the_placeholder_keeps_its_volume() -> None:
    assert calculate_unit_cell_volume(1, 1, 2, 90, 90, 90) == pytest.approx(2.0)


def test_atom_site_items_become_columns_and_quoted_placeholders_stay_text() -> None:
    columns, atom_count = expand_mmcif_atom_fields(
        {
            "_atom_site.id": ["1", "2", "3"],
            # The reader gives a bare placeholder as None and a quoted one as
            # the text it is in CIF.
            "_atom_site.pdbx_PDB_ins_code": [None, "?", "A"],
        }
    )

    assert atom_count == 3
    assert columns == {
        "_atom_site.id": ["1", "2", "3"],
        "_atom_site.pdbx_PDB_ins_code": [None, "?", "A"],
    }


def test_one_row_category_is_repeated_on_every_atom() -> None:
    columns, _ = expand_mmcif_atom_fields(
        {
            "_atom_site.id": ["1", "2"],
            "_exptl.method": ["X-RAY DIFFRACTION"],
            "_refine.ls_d_res_high": [None],
        }
    )

    assert columns["_exptl.method"] == ["X-RAY DIFFRACTION", "X-RAY DIFFRACTION"]
    assert columns["_refine.ls_d_res_high"] == [None, None]


def test_multi_row_category_is_left_out_even_when_it_has_one_row_per_atom() -> None:
    columns, _ = expand_mmcif_atom_fields(
        {
            "_atom_site.id": ["1", "2"],
            "_entity.id": ["1", "2"],
            "_entity.type": ["polymer", "water"],
        }
    )

    assert set(columns) == {"_atom_site.id"}


def test_one_row_anisotropic_loop_is_not_repeated_on_every_atom() -> None:
    columns, _ = expand_mmcif_atom_fields(
        {
            "_atom_site.id": ["1", "2"],
            "_atom_site_anisotrop.id": ["1"],
            "_atom_site_anisotrop.U[1][1]": ["0.1"],
        }
    )

    assert set(columns) == {"_atom_site.id"}


def test_line_breaks_in_text_values_become_spaces() -> None:
    columns, _ = expand_mmcif_atom_fields(
        {
            "_atom_site.id": ["1"],
            "_struct.title": ["First line\r\nsecond\rthird\nfourth\n"],
        }
    )

    assert columns["_struct.title"] == ["First line second third fourth"]


def test_values_without_line_breaks_are_kept_as_written() -> None:
    columns, _ = expand_mmcif_atom_fields(
        {"_atom_site.id": ["1"], "_symmetry.space_group_name_H-M": [" P 1 "]}
    )

    assert columns["_symmetry.space_group_name_H-M"] == [" P 1 "]


def test_dictionary_without_atom_site_gives_no_columns() -> None:
    assert expand_mmcif_atom_fields({"_exptl.method": ["SOLUTION NMR"]}) == ({}, 0)


def test_empty_atom_site_loop_gives_no_columns() -> None:
    assert expand_mmcif_atom_fields(
        {"_atom_site.id": [], "_exptl.method": ["X-RAY DIFFRACTION"]}
    ) == ({}, 0)
