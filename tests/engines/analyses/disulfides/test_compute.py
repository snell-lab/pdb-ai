from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from pathlib import Path

import pandas as pd
import pytest

from pdbai.engines.analyses.disulfides.columns import DISULFIDES_TABLE
from pdbai.engines.analyses.disulfides.compute import (
    DisulfideComputation,
    compute_structure_disulfides,
)
from pdbai.engines.core import site_key_dataframe
from pdbai.engines.core.tables import validate_tables
from pdbai.structure import parse_mmcif_atom_rows, read_mmcif_as_flat_dict

type FlatMmcif = dict[str, list[str | None]]

_ATOM_ITEMS = (
    "group_PDB",
    "id",
    "type_symbol",
    "label_atom_id",
    "label_alt_id",
    "label_comp_id",
    "label_asym_id",
    "label_entity_id",
    "label_seq_id",
    "pdbx_PDB_ins_code",
    "Cartn_x",
    "Cartn_y",
    "Cartn_z",
    "occupancy",
    "B_iso_or_equiv",
    "auth_seq_id",
    "auth_comp_id",
    "auth_asym_id",
    "auth_atom_id",
    "pdbx_PDB_model_num",
)


def _atom(
    res_num: str,
    atom: str = "SG",
    *,
    x: float | str = 0.0,
    y: float | str = 0.0,
    z: float | str = 0.0,
    altloc: str = ".",
    occupancy: str = "1.00",
    b_factor: str = "20",
    chain: str = "A",
    ins_code: str = "?",
    model: str = "1",
    residue: str = "CYS",
    label_seq: str | None = None,
    element: str | None = None,
) -> dict[str, str]:
    """Return one _atom_site row, a cysteine SG at the origin by default.

    The label sequence number is the author number unless given. The element
    is the first letter of a cysteine's atom name, and the whole atom name of
    any other residue's, unless given.
    """
    if element is None:
        element = atom[0] if residue == "CYS" else atom
    return {
        "group_PDB": "ATOM" if residue == "CYS" else "HETATM",
        "type_symbol": element,
        "label_atom_id": atom,
        "label_alt_id": altloc,
        "label_comp_id": residue,
        "label_asym_id": chain,
        "label_entity_id": "1",
        "label_seq_id": res_num if label_seq is None else label_seq,
        "pdbx_PDB_ins_code": ins_code,
        "Cartn_x": str(x),
        "Cartn_y": str(y),
        "Cartn_z": str(z),
        "occupancy": occupancy,
        "B_iso_or_equiv": b_factor,
        "auth_seq_id": res_num,
        "auth_comp_id": residue,
        "auth_asym_id": chain,
        "auth_atom_id": atom,
        "pdbx_PDB_model_num": model,
    }


def _cysteine(
    res_num: str,
    x: float = 0.0,
    y: float = 0.0,
    z: float = 0.0,
    *,
    sulfur_b_factor: str = "20",
    carbon_b_factor: str = "18",
    **items: str,
) -> list[dict[str, str]]:
    """Return a cysteine's CB and SG rows, its SG at the point given."""
    return [
        _atom(res_num, "CB", x=x, y=y, z=z - 1.8, b_factor=carbon_b_factor, **items),
        _atom(res_num, "SG", x=x, y=y, z=z, b_factor=sulfur_b_factor, **items),
    ]


def _record(
    first: str, second: str, record_id: str = "disulf1", **items: str | None
) -> dict[str, str | None]:
    """Return a disulf record joining the SG atoms of two chain A cysteines."""
    return {
        "id": record_id,
        "conn_type_id": "disulf",
        "ptnr1_label_comp_id": "CYS",
        "ptnr1_label_atom_id": "SG",
        "ptnr1_auth_asym_id": "A",
        "ptnr1_auth_seq_id": first,
        "ptnr2_label_comp_id": "CYS",
        "ptnr2_label_atom_id": "SG",
        "ptnr2_auth_asym_id": "A",
        "ptnr2_auth_seq_id": second,
        **items,
    }


def _write(
    path: Path,
    atoms: Sequence[Mapping[str, str]],
    records: Sequence[Mapping[str, str | None]] = (),
    *,
    omit: Sequence[str] = (),
) -> None:
    """Write the atoms, numbered in order, and the records as an mmCIF file."""
    items = [item for item in _ATOM_ITEMS if item not in omit]
    lines = ["data_1abc", "loop_", *(f"_atom_site.{item}" for item in items)]
    lines += [
        " ".join(str(number) if item == "id" else atom[item] for item in items)
        for number, atom in enumerate(atoms, start=1)
    ]
    if records:
        record_items = list(
            dict.fromkeys(item for record in records for item in record)
        )
        lines += ["", "loop_", *(f"_struct_conn.{item}" for item in record_items)]
        lines += [
            " ".join(record.get(item) or "?" for item in record_items)
            for record in records
        ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _read(path: Path) -> tuple[pd.DataFrame, FlatMmcif]:
    """Return a file's parsed atom table and its flat dictionary."""
    atoms = parse_mmcif_atom_rows(path, {}, structure_id=path.stem).dataframe
    return atoms, read_mmcif_as_flat_dict(path)


def _structure(
    tmp_path: Path,
    atoms: Sequence[Mapping[str, str]],
    records: Sequence[Mapping[str, str | None]] = (),
    *,
    omit: Sequence[str] = (),
) -> tuple[pd.DataFrame, FlatMmcif]:
    """Write a structure of entry 1ABC and return its atom table and dictionary."""
    path = tmp_path / "1abc.cif"
    _write(path, atoms, records, omit=omit)
    return _read(path)


def _compute(
    tmp_path: Path,
    atoms: Sequence[Mapping[str, str]],
    records: Sequence[Mapping[str, str | None]] = (),
    *,
    omit: Sequence[str] = (),
) -> DisulfideComputation:
    """Return the disulfides of a structure written from atoms and records."""
    atom_table, mmcif = _structure(tmp_path, atoms, records, omit=omit)
    return compute_structure_disulfides(atom_table, mmcif, pdb_id="1abc")


def _fixture(path: Path) -> tuple[pd.DataFrame, DisulfideComputation]:
    """Return the fixture's atom table and its disulfides."""
    atoms, mmcif = _read(path)
    return atoms, compute_structure_disulfides(atoms, mmcif, pdb_id="1dsf")


# The fixture's rows as its docstring works them out: the two residues, the
# detection method, the distances, the two occupancies, the alternates flag,
# the two ratios, and the status.
_EXPECTED_ROWS = [
    (
        "1DSF:1:A:101::SG:",
        "1DSF:1:A:140::SG:",
        "struct_conn_and_distance",
        2.04,
        2.04,
        1.0,
        1.0,
        False,
        20 / 18,
        20 / 18,
        "ok",
    ),
    (
        "1DSF:1:A:105::SG:A",
        "1DSF:1:A:130::SG:A",
        "struct_conn_and_distance",
        2.05,
        3.6,
        0.4,
        1.0,
        True,
        1.25,
        1.5,
        "ok",
    ),
    (
        "1DSF:1:A:120:A:SG:",
        "1DSF:1:A:160::SG:",
        "distance",
        2.7,
        2.7,
        1.0,
        1.0,
        False,
        1.2,
        1.5,
        "ok",
    ),
    (
        "1DSF:1:A:125::SG:",
        "1DSF:1:B:301::SG:",
        "struct_conn_and_distance",
        2.03,
        2.03,
        1.0,
        1.0,
        False,
        math.nan,
        math.nan,
        "b_factor_unavailable",
    ),
    (
        "1DSF:1:A:150::SG:A",
        "1DSF:1:A:155::SG:",
        "distance",
        2.05,
        2.05,
        0.45,
        0.45,
        False,
        1.1,
        1.5,
        "sg_not_in_atom_table",
    ),
    (
        "1DSF:1:A:176::SG:",
        "1DSF:1:A:194::SG:",
        "struct_conn",
        3.6,
        3.6,
        0.7,
        0.7,
        False,
        1.0,
        1.25,
        "ok",
    ),
    (
        "1DSF:1:A:210::SG:",
        "1DSF:1:A:213::SG:",
        "distance",
        2.9,
        2.9,
        1.0,
        1.0,
        False,
        1.1,
        1.2,
        "ok",
    ),
    (
        "1DSF:2:A:101::SG:",
        "1DSF:2:A:140::SG:",
        "struct_conn_and_distance",
        2.6,
        2.6,
        1.0,
        1.0,
        False,
        math.nan,
        math.nan,
        "b_factor_unavailable",
    ),
]

_MEASURE_COLUMNS = [
    "site_id",
    "partner_site_id",
    "detection_method",
    "ss_distance_min_A",
    "ss_distance_max_A",
    "sg_conformer_occupancy_min",
    "sg_total_occupancy_min",
    "sg_has_alternate_conformers",
    "sg_cb_b_factor_ratio_min",
    "sg_cb_b_factor_ratio_max",
    "pair_status",
]


def _approx_row(row: Sequence[object]) -> list[object]:
    """Return a row with its numbers compared to 1e-9 and NaN equal to NaN."""
    return [
        pytest.approx(value, abs=1e-9, nan_ok=True)
        if isinstance(value, float)
        else value
        for value in row
    ]


def test_the_fixture_gives_the_rows_its_docstring_works_out(
    disulfide_cif: Path,
) -> None:
    _atoms, computation = _fixture(disulfide_cif)

    table = computation.table
    assert list(table.columns) == list(DISULFIDES_TABLE.column_names)
    for column in DISULFIDES_TABLE.columns:
        assert table[column.name].dtype == column.dtype, column.name
    rows = [list(row) for row in table[_MEASURE_COLUMNS].itertuples(index=False)]
    assert rows == [_approx_row(row) for row in _EXPECTED_ROWS]
    assert len(table.columns) == 29
    assert list(table.columns).index("metal_bridged") == (
        list(table.columns).index("detection_method") + 1
    )
    assert table["model_num"].tolist() == [1, 1, 1, 1, 1, 1, 1, 2]
    assert table["partner_model_num"].tolist() == table["model_num"].tolist()
    assert set(table["pdb_id"]) == set(table["partner_pdb_id"]) == {"1DSF"}
    assert set(table["res_name"]) == set(table["partner_res_name"]) == {"CYS"}
    assert set(table["atom_name"]) == set(table["partner_atom_name"]) == {"SG"}
    assert table["ins_code"].tolist() == ["", "", "A", "", "", "", "", ""]
    assert table["altloc"].tolist() == ["", "A", "", "", "A", "", "", ""]
    assert table["partner_chain_id"].tolist()[3] == "B"
    assert table["sg_occupancy_deficit_max"].tolist() == pytest.approx(
        [0.0, 0.0, 0.0, 0.0, 0.55, 0.3, 0.0, 0.0], abs=1e-9
    )
    # Only Cys 210 and Cys 213 share a zinc. The flag is never missing.
    assert table["metal_bridged"].tolist() == [
        False,
        False,
        False,
        False,
        False,
        False,
        True,
        False,
    ]
    assert computation.status == "degraded"
    assert computation.unmatched_record_count == 2
    assert computation.symmetry_record_count == 1
    # Cys 195 has a CB and no SG.
    assert computation.cysteine_without_sg_count == 1
    # Records that match in model 1 only, such as disulf2, raise no warning.
    assert computation.warnings == (
        "_struct_conn record 'disulf4' names a cysteine that has no SG atom, so "
        "it gives no row.",
        "_struct_conn record 'disulf5' names a cysteine with no SG or CB atom in "
        "the structure, so it gives no row.",
        "cysteine A 150 in model 1 has no SG in the conformer the atom table "
        "keeps, so its rows name SG conformer A and no atom links to it.",
    )


def test_every_conformer_is_read_and_the_kept_one_names_the_residue(
    disulfide_cif: Path,
) -> None:
    atoms, computation = _fixture(disulfide_cif)

    kept_sulfur = atoms[atoms["res_num"].eq("105") & atoms["atom_name"].eq("SG")]
    # The atom table holds conformer A alone, yet the row's longest distance
    # is that of conformer B.
    assert kept_sulfur["_atom_site.label_alt_id"].tolist() == ["A"]
    row = computation.table.iloc[1]
    assert row["site_id"] == site_key_dataframe(kept_sulfur)["site_id"].item()
    assert row["ss_distance_max_A"] == pytest.approx(3.6, abs=1e-9)


def test_a_residue_whose_kept_conformer_has_no_sg_falls_back_to_its_best_sg(
    disulfide_cif: Path,
) -> None:
    atoms, computation = _fixture(disulfide_cif)

    residue_150 = atoms[atoms["res_num"].eq("150")]
    assert residue_150["atom_name"].tolist() == ["CB"]
    row = computation.table.iloc[4]
    assert (row["site_id"], row["altloc"], row["pair_status"]) == (
        "1DSF:1:A:150::SG:A",
        "A",
        "sg_not_in_atom_table",
    )
    # CB A, which can coexist with SG A, gives 30 / 20. The kept CB B would
    # give 30 / 25.
    assert row["sg_cb_b_factor_ratio_max"] == pytest.approx(1.5, abs=1e-9)
    assert not computation.link.loc[residue_150.index].notna().any()
    assert computation.link.dropna().tolist() == [
        "1DSF:1:A:101::SG:",
        "1DSF:1:A:140::SG:",
        "1DSF:1:A:105::SG:A",
        "1DSF:1:A:130::SG:A",
        "1DSF:1:A:120:A:SG:",
        "1DSF:1:A:160::SG:",
        "1DSF:1:A:125::SG:",
        "1DSF:1:B:301::SG:",
        "1DSF:1:A:155::SG:",
        "1DSF:1:A:176::SG:",
        "1DSF:1:A:194::SG:",
        "1DSF:1:A:210::SG:",
        "1DSF:1:A:213::SG:",
        "1DSF:2:A:101::SG:",
        "1DSF:2:A:140::SG:",
    ]


def test_a_zinc_site_gives_no_rows(disulfide_cif: Path) -> None:
    _atoms, computation = _fixture(disulfide_cif)

    zinc_ligands = {"170", "173", "186", "189"}
    table = computation.table
    assert not table["res_num"].isin(zinc_ligands).any()
    assert not table["partner_res_num"].isin(zinc_ligands).any()


@pytest.mark.parametrize(
    ("distance", "found"),
    [
        (3.0, True),
        (3.001, False),
        (2.04, True),
        # The search has no lower limit.
        (1.5, True),
        (1.499, True),
        (0.867, True),
        (0.001, True),
    ],
)
def test_the_search_includes_its_limit_and_has_no_lower_one(
    tmp_path: Path, distance: float, found: bool
) -> None:
    computation = _compute(tmp_path, [*_cysteine("1"), *_cysteine("2", x=distance)])

    assert len(computation.table) == int(found)


def test_a_pair_below_bonding_distance_is_reported_as_measured(
    tmp_path: Path,
) -> None:
    # 1.160 Å, the distance of the annotated bond Cys 50 to Cys 133 of 1OYF.
    atoms = [*_cysteine("1"), *_cysteine("2", x=1.16)]

    by_distance = _compute(tmp_path, atoms).table.iloc[0]
    by_both = _compute(tmp_path, atoms, [_record("1", "2")]).table.iloc[0]

    assert by_distance["detection_method"] == "distance"
    assert by_both["detection_method"] == "struct_conn_and_distance"
    for row in (by_distance, by_both):
        assert row["ss_distance_min_A"] == pytest.approx(1.16, abs=1e-9)
        assert row["ss_distance_max_A"] == pytest.approx(1.16, abs=1e-9)
        assert row["pair_status"] == "ok"


def test_two_sulfurs_at_one_point_are_a_row_at_zero_distance(tmp_path: Path) -> None:
    computation = _compute(tmp_path, [*_cysteine("1"), *_cysteine("1", chain="B")])

    row = computation.table.iloc[0]
    assert (row["site_id"], row["partner_site_id"]) == (
        "1ABC:1:A:1::SG:",
        "1ABC:1:B:1::SG:",
    )
    assert row["ss_distance_min_A"] == 0.0


def test_the_limit_off_an_axis_is_included_despite_rounding_error(
    tmp_path: Path,
) -> None:
    # 3.000 Å in decimals, 3.0000000000000018 in floating point.
    computation = _compute(
        tmp_path,
        [*_cysteine("1", 13.7, 0.1, 9.9), *_cysteine("2", 16.1, 1.9, 9.9)],
    )

    assert computation.table["detection_method"].tolist() == ["distance"]


_A_ONLY_AGAINST_B_ONLY = [
    *_cysteine("1", altloc="A", occupancy="0.60"),
    *_cysteine("2", x=2.0, altloc="B", occupancy="0.40"),
]


def test_conformers_that_cannot_coexist_are_never_paired(tmp_path: Path) -> None:
    assert _compute(tmp_path, _A_ONLY_AGAINST_B_ONLY).table.empty


def test_conformers_that_cannot_coexist_are_never_measured(tmp_path: Path) -> None:
    row = _compute(tmp_path, _A_ONLY_AGAINST_B_ONLY, [_record("1", "2")]).table.iloc[0]
    assert row["detection_method"] == "struct_conn"
    assert math.isnan(row["ss_distance_min_A"])
    assert math.isnan(row["ss_distance_max_A"])
    assert row["pair_status"] == "no_compatible_conformers"


def test_an_unlabelled_sg_pairs_with_every_labelled_one(tmp_path: Path) -> None:
    atoms = [
        *_cysteine("1"),
        _atom("2", "CB", x=2.05, z=-1.8),
        _atom("2", "SG", x=2.05, altloc="A", occupancy="0.60"),
        _atom("2", "SG", y=2.5, altloc="B", occupancy="0.40"),
    ]

    row = _compute(tmp_path, atoms).table.iloc[0]

    assert row["ss_distance_min_A"] == pytest.approx(2.05, abs=1e-9)
    assert row["ss_distance_max_A"] == pytest.approx(2.5, abs=1e-9)
    assert row["sg_has_alternate_conformers"]


def test_a_zero_occupancy_sg_is_not_searched_but_counts_in_occupancy(
    tmp_path: Path,
) -> None:
    atoms = [*_cysteine("1", occupancy="0.00"), *_cysteine("2", x=2.04)]

    assert _compute(tmp_path, atoms).table.empty
    row = _compute(tmp_path, atoms, [_record("1", "2")]).table.iloc[0]
    assert math.isnan(row["ss_distance_min_A"])
    assert row["sg_conformer_occupancy_min"] == 0.0
    assert row["sg_total_occupancy_min"] == 0.0
    assert row["pair_status"] == "no_usable_sg"


def test_a_zero_occupancy_residue_gives_no_b_factor_ratio(tmp_path: Path) -> None:
    atoms = [*_cysteine("1", occupancy="0.00"), *_cysteine("2", x=2.04)]

    row = _compute(tmp_path, atoms, [_record("1", "2")]).table.iloc[0]

    # Both atoms of residue 1 carry a B-factor of 20, which was never fitted.
    assert math.isnan(row["sg_cb_b_factor_ratio_min"])
    assert math.isnan(row["sg_cb_b_factor_ratio_max"])
    # The deficit is plain arithmetic on the deposited occupancies.
    assert row["sg_occupancy_deficit_max"] == 0.0


def test_a_zero_occupancy_cb_gives_no_b_factor_ratio(tmp_path: Path) -> None:
    atoms = [
        _atom("1", "CB", z=-1.8, occupancy="0.00"),
        _atom("1"),
        *_cysteine("2", x=2.04),
    ]

    row = _compute(tmp_path, atoms).table.iloc[0]

    assert math.isnan(row["sg_cb_b_factor_ratio_min"])
    assert math.isnan(row["sg_cb_b_factor_ratio_max"])
    assert row["pair_status"] == "b_factor_unavailable"


def test_an_sg_without_coordinates_is_not_usable(tmp_path: Path) -> None:
    atoms = [
        _atom("1", "CB", x="?", y="?", z="?"),
        _atom("1", x="?", y="?", z="?"),
        *_cysteine("2", x=2.04),
        *_cysteine("3", x=20.0),
        *_cysteine("4", x=22.04),
    ]

    computation = _compute(tmp_path, atoms, [_record("1", "2")])

    assert computation.table["pair_status"].tolist() == ["no_usable_sg", "ok"]
    assert computation.table["detection_method"].tolist() == [
        "struct_conn",
        "distance",
    ]


def test_an_sg_with_a_missing_occupancy_is_searched(tmp_path: Path) -> None:
    computation = _compute(
        tmp_path, [*_cysteine("1", occupancy="?"), *_cysteine("2", x=2.04)]
    )

    row = computation.table.iloc[0]
    assert row["ss_distance_min_A"] == pytest.approx(2.04, abs=1e-9)
    assert math.isnan(row["sg_conformer_occupancy_min"])
    assert math.isnan(row["sg_total_occupancy_min"])
    assert row["pair_status"] == "occupancy_missing"


def test_the_search_never_crosses_models(tmp_path: Path) -> None:
    atoms = [*_cysteine("1"), *_cysteine("2", x=2.04, model="2")]

    assert _compute(tmp_path, atoms).table.empty


_BONDED_PAIR = [*_cysteine("1"), *_cysteine("2", x=2.04)]


def test_cysteines_are_found_by_label_name_when_the_file_has_no_author_name(
    tmp_path: Path,
) -> None:
    atom_table, mmcif = _structure(
        tmp_path, _BONDED_PAIR, [_record("1", "2")], omit=("auth_comp_id",)
    )

    computation = compute_structure_disulfides(atom_table, mmcif, pdb_id="1abc")

    row = computation.table.iloc[0]
    assert row["detection_method"] == "struct_conn_and_distance"
    assert row["ss_distance_min_A"] == pytest.approx(2.04, abs=1e-9)
    assert row["pair_status"] == "ok"
    # The key columns hold what the atom table holds, and the parser names
    # these atoms by their label component.
    assert set(atom_table["res_name"]) == {"CYS"}
    assert (row["res_name"], row["partner_res_name"]) == ("CYS", "CYS")
    assert (row["site_id"], row["partner_site_id"]) == (
        "1ABC:1:A:1::SG:",
        "1ABC:1:A:2::SG:",
    )
    # The atom table names its cysteines by the same rule, so the rows the
    # parser kept are matched and both SG atoms link.
    assert computation.link.dropna().tolist() == [
        "1ABC:1:A:1::SG:",
        "1ABC:1:A:2::SG:",
    ]
    assert computation.status == "ok"
    assert computation.warnings == ()


def test_the_label_name_is_used_row_by_row(tmp_path: Path) -> None:
    # Cys 1 has no author name on either atom, and Cys 2 has one.
    atoms = [
        *({**atom, "auth_comp_id": "?"} for atom in _cysteine("1")),
        *_cysteine("2", x=2.04),
    ]

    row = _compute(tmp_path, atoms).table.iloc[0]

    assert row["ss_distance_min_A"] == pytest.approx(2.04, abs=1e-9)
    assert (row["res_name"], row["partner_res_name"]) == ("CYS", "CYS")
    assert row["pair_status"] == "ok"


@pytest.mark.parametrize(
    ("author", "label", "found"),
    [
        ("CYS", "CSO", True),
        ("CSO", "CYS", False),
        ("cys", "CYS", True),
        ("?", "cys", True),
        ("?", "CSO", False),
        ("?", "?", False),
    ],
)
def test_the_author_name_decides_whenever_it_is_given(
    tmp_path: Path, author: str, label: str, found: bool
) -> None:
    second = [
        {**atom, "auth_comp_id": author, "label_comp_id": label}
        for atom in _cysteine("2", x=2.04)
    ]

    computation = _compute(tmp_path, [*_cysteine("1"), *second])

    assert len(computation.table) == int(found)
    assert computation.cysteine_without_sg_count == 0


def test_a_file_with_neither_residue_name_gives_no_rows(tmp_path: Path) -> None:
    computation = _compute(
        tmp_path, _BONDED_PAIR, omit=("auth_comp_id", "label_comp_id")
    )

    assert computation.table.empty
    assert computation.status == "ok"


def test_a_file_without_author_residue_numbers_gives_rows_under_label_numbering(
    tmp_path: Path,
) -> None:
    # Without auth_seq_id the whole file is numbered by its label fields, so
    # the cysteines of author chain A are told apart by label chain L and
    # their label numbers, as the atom table names them.
    atoms = [
        *({**atom, "label_asym_id": "L"} for atom in _cysteine("1", label_seq="11")),
        *(
            {**atom, "label_asym_id": "L"}
            for atom in _cysteine("2", x=2.04, label_seq="12")
        ),
    ]
    atom_table, mmcif = _structure(
        tmp_path, atoms, omit=("auth_comp_id", "auth_seq_id")
    )

    computation = compute_structure_disulfides(atom_table, mmcif, pdb_id="1abc")

    row = computation.table.iloc[0]
    assert set(atom_table["residue_numbering"]) == {"label"}
    assert (row["site_id"], row["partner_site_id"]) == (
        "1ABC:1:L:11::SG:",
        "1ABC:1:L:12::SG:",
    )
    assert (row["chain_id"], row["res_num"], row["res_name"]) == ("L", "11", "CYS")
    assert row["detection_method"] == "distance"
    assert row["pair_status"] == "ok"
    assert computation.link.dropna().tolist() == [
        "1ABC:1:L:11::SG:",
        "1ABC:1:L:12::SG:",
    ]
    assert computation.status == "ok"


def test_one_cysteine_without_author_residue_numbers_is_no_failure(
    tmp_path: Path,
) -> None:
    computation = _compute(
        tmp_path, _cysteine("1"), omit=("auth_comp_id", "auth_seq_id")
    )

    assert computation.table.empty
    assert computation.status == "ok"


def test_the_atom_table_is_checked_through_the_label_name_too(tmp_path: Path) -> None:
    atom_table, mmcif = _structure(tmp_path, _BONDED_PAIR, omit=("auth_comp_id",))
    renamed = atom_table.copy()
    renamed.loc[1, "chain_id"] = "Z"
    extra = pd.concat(
        [atom_table, atom_table.iloc[[1]].assign(**{"_atom_site.id": "99"})]
    ).reset_index(drop=True)

    with pytest.raises(RuntimeError, match="does not match the atom table"):
        compute_structure_disulfides(renamed, mmcif, pdb_id="1abc")
    with pytest.raises(RuntimeError, match="holds no atom for"):
        compute_structure_disulfides(extra, mmcif, pdb_id="1abc")


def _zinc(
    x: float | str = 0.0, y: float | str = 0.0, z: float | str = 0.0, **items: str
) -> dict[str, str]:
    """Return a zinc ion, residue A 401, at the point given."""
    return _atom("401", "ZN", x=x, y=y, z=z, residue="ZN", label_seq=".", **items)


# Two cysteines whose SG atoms lie 2.9 Å apart and 2.330 Å from the origin,
# since sqrt(1.45² + 1.824²) = 2.330. Two ligands of an intact tetrahedral
# zinc site would lie about 3.8 Å apart.
_PAIR_AROUND_ORIGIN = [
    *_cysteine("1", x=-1.45, y=1.824),
    *_cysteine("2", x=1.45, y=1.824),
]


def test_a_short_sg_pair_beside_a_zinc_is_kept_and_flagged(tmp_path: Path) -> None:
    metal_record = {
        "id": "metalc1",
        "conn_type_id": "metalc",
        "ptnr1_auth_asym_id": "A",
        "ptnr1_auth_seq_id": "401",
        "ptnr1_label_comp_id": "ZN",
        "ptnr2_auth_asym_id": "A",
        "ptnr2_auth_seq_id": "1",
        "ptnr2_label_comp_id": "CYS",
    }

    with_record = _compute(tmp_path, [_zinc(), *_PAIR_AROUND_ORIGIN], [metal_record])
    without_record = _compute(tmp_path, [_zinc(), *_PAIR_AROUND_ORIGIN])
    without_zinc = _compute(tmp_path, _PAIR_AROUND_ORIGIN)

    row = with_record.table.iloc[0]
    assert row["detection_method"] == "distance"
    assert row["ss_distance_min_A"] == pytest.approx(2.9, abs=1e-9)
    assert row["metal_bridged"]
    # The flag comes from the coordinates and reads no metalc record.
    pd.testing.assert_frame_equal(with_record.table, without_record.table)
    # The flag is information. It changes no other value of the row.
    assert not without_zinc.table["metal_bridged"].any()
    pd.testing.assert_frame_equal(
        with_record.table.drop(columns="metal_bridged"),
        without_zinc.table.drop(columns="metal_bridged"),
    )
    assert with_record.link.dropna().tolist() == without_zinc.link.dropna().tolist()


def test_a_pair_with_one_sg_on_the_metal_is_not_flagged(tmp_path: Path) -> None:
    # The zinc is 2.33 Å from SG 1 and 2.9 + 2.33 = 5.23 Å from SG 2.
    atoms = [_zinc(x=-2.33), *_cysteine("1"), *_cysteine("2", x=2.9)]

    table = _compute(tmp_path, atoms).table

    assert table["ss_distance_min_A"].tolist() == pytest.approx([2.9], abs=1e-9)
    assert table["metal_bridged"].tolist() == [False]


def test_two_sulfurs_on_different_metal_atoms_are_not_flagged(tmp_path: Path) -> None:
    atoms = [
        _zinc(x=-2.33),
        _atom("402", "ZN", x=5.23, residue="ZN", label_seq="."),
        *_cysteine("1"),
        *_cysteine("2", x=2.9),
    ]

    assert _compute(tmp_path, atoms).table["metal_bridged"].tolist() == [False]


@pytest.mark.parametrize(
    ("zinc_altloc", "bridged"),
    [
        # An unlabelled zinc can coexist with every conformer, and conformer B
        # of Cys 1 and the SG of Cys 2 are both bound to it.
        (".", True),
        ("B", True),
        # Conformer A of Cys 1 is 14 Å from the zinc, and conformer B cannot
        # coexist with a zinc labelled A.
        ("A", False),
    ],
)
def test_the_flag_needs_conformers_that_can_coexist_with_the_metal(
    tmp_path: Path, zinc_altloc: str, bridged: bool
) -> None:
    atoms = [
        _zinc(altloc=zinc_altloc, occupancy="0.50"),
        _atom("1", "CB", x=-1.45, y=1.824, z=-1.8),
        _atom("1", x=-1.45, y=14.0, altloc="A", occupancy="0.50"),
        _atom("1", x=-1.45, y=1.824, altloc="B", occupancy="0.50"),
        *_cysteine("2", x=1.45, y=1.824),
    ]

    row = _compute(tmp_path, atoms).table.iloc[0]

    assert row["ss_distance_min_A"] == pytest.approx(2.9, abs=1e-9)
    assert bool(row["metal_bridged"]) is bridged


def test_the_flag_needs_sulfur_conformers_that_can_coexist(tmp_path: Path) -> None:
    # Conformer A of Cys 1 and conformer B of Cys 2 are each bound to the
    # zinc, yet they cannot coexist. The row comes from the two A conformers,
    # of which only that of Cys 1 is near the zinc.
    atoms = [
        _zinc(),
        *_cysteine("1", x=-1.45, y=1.824, altloc="A", occupancy="0.50"),
        *_cysteine("2", x=-1.45, y=4.724, altloc="A", occupancy="0.50"),
        *_cysteine("2", x=1.45, y=1.824, altloc="B", occupancy="0.50"),
    ]

    row = _compute(tmp_path, atoms).table.iloc[0]

    assert row["ss_distance_min_A"] == pytest.approx(2.9, abs=1e-9)
    assert row["ss_distance_max_A"] == pytest.approx(2.9, abs=1e-9)
    assert not row["metal_bridged"]


@pytest.mark.parametrize(
    ("separation", "bridged"), [(2.8, True), (2.801, False), (2.33, True)]
)
def test_the_metal_contact_distance_includes_its_limit(
    tmp_path: Path, separation: float, bridged: bool
) -> None:
    # Both sulfurs are at the same distance from the zinc, on either side of
    # it along y, and the record makes them a row whatever lies between them.
    atoms = [
        _zinc(),
        *_cysteine("1", y=separation),
        *_cysteine("2", y=-separation),
    ]

    table = _compute(tmp_path, atoms, [_record("1", "2")]).table

    assert table["metal_bridged"].tolist() == [bridged]


def test_a_metal_contact_off_an_axis_is_included_despite_rounding_error(
    tmp_path: Path,
) -> None:
    # Each sulfur is 2.800 Å from the zinc in decimals and 2.8000000000000003
    # in floating point.
    atoms = [
        _zinc(x=7.0, y=6.9, z=6.6),
        *_cysteine("1", 9.4, 8.1, 7.4),
        *_cysteine("2", 4.6, 5.7, 5.8),
    ]

    table = _compute(tmp_path, atoms, [_record("1", "2")]).table

    assert table["metal_bridged"].tolist() == [True]


@pytest.mark.parametrize(
    ("element", "bridged"),
    [
        ("ZN", True),
        ("Zn", True),
        ("zn", True),
        ("ZN2+", True),
        ("Zn2+", True),
        ("FE", True),
        ("Fe3+", True),
        ("NI", True),
        # gemmi does not read a one-letter symbol before a digit, so the
        # engine hands it the letters alone.
        ("K1+", True),
        # Sulfur, selenium, and hydrogen and its isotope are not metals.
        ("S", False),
        ("SE", False),
        ("H", False),
        ("D", False),
        # Unknown elements and unreadable symbols are not metals.
        ("X", False),
        ("?", False),
        ("ZNX", False),
        ("2+", False),
    ],
)
def test_a_metal_is_an_atom_whose_type_symbol_names_a_metallic_element(
    tmp_path: Path, element: str, bridged: bool
) -> None:
    atoms = [_zinc(element=element), *_PAIR_AROUND_ORIGIN]

    assert _compute(tmp_path, atoms).table["metal_bridged"].tolist() == [bridged]


def test_a_metal_of_any_residue_and_record_type_counts(tmp_path: Path) -> None:
    # The iron of a cluster residue recorded as ATOM, named FE1 as in FES.
    iron = {
        **_atom("501", "FE1", residue="FES", label_seq=".", element="FE"),
        "group_PDB": "ATOM",
    }

    table = _compute(tmp_path, [iron, *_PAIR_AROUND_ORIGIN]).table

    assert table["metal_bridged"].tolist() == [True]


def test_a_file_that_names_no_elements_flags_nothing(tmp_path: Path) -> None:
    computation = _compute(
        tmp_path, [_zinc(), *_PAIR_AROUND_ORIGIN], omit=("type_symbol",)
    )

    assert computation.table["metal_bridged"].tolist() == [False]
    assert computation.table["metal_bridged"].dtype == "boolean"
    assert computation.warnings == ()


@pytest.mark.parametrize(
    "zinc",
    [
        _zinc(occupancy="0.00"),
        _zinc(occupancy="-1.00"),
        _zinc(x="?", y="?", z="?"),
        _zinc(model="2"),
    ],
    ids=["zero_occupancy", "negative_occupancy", "no_coordinates", "another_model"],
)
def test_an_unusable_metal_or_one_of_another_model_flags_nothing(
    tmp_path: Path, zinc: dict[str, str]
) -> None:
    table = _compute(tmp_path, [*_PAIR_AROUND_ORIGIN, zinc]).table

    assert table["metal_bridged"].tolist() == [False]


def test_a_metal_with_a_missing_occupancy_is_usable(tmp_path: Path) -> None:
    table = _compute(tmp_path, [_zinc(occupancy="?"), *_PAIR_AROUND_ORIGIN]).table

    assert table["metal_bridged"].tolist() == [True]


def test_the_flag_is_set_model_by_model(tmp_path: Path) -> None:
    atoms = [
        _zinc(),
        *_PAIR_AROUND_ORIGIN,
        *_cysteine("1", x=-1.45, y=1.824, model="2"),
        *_cysteine("2", x=1.45, y=1.824, model="2"),
        _zinc(y=20.0, model="2"),
    ]

    table = _compute(tmp_path, atoms).table

    assert table["model_num"].tolist() == [1, 2]
    assert table["metal_bridged"].tolist() == [True, False]


def test_a_zero_occupancy_sg_on_a_metal_does_not_bridge(tmp_path: Path) -> None:
    atoms = [
        _zinc(),
        *_cysteine("1", x=-1.45, y=1.824, occupancy="0.00"),
        *_cysteine("2", x=1.45, y=1.824),
    ]

    row = _compute(tmp_path, atoms, [_record("1", "2")]).table.iloc[0]

    assert row["pair_status"] == "no_usable_sg"
    assert not row["metal_bridged"]


def test_reversed_per_conformer_and_repeated_records_give_one_row(
    tmp_path: Path,
) -> None:
    atoms = [*_cysteine("1"), *_cysteine("2", x=2.04)]
    records = [
        _record("1", "2", "disulf1", pdbx_ptnr1_label_alt_id="A"),
        _record("2", "1", "disulf2", pdbx_ptnr1_label_alt_id="B"),
        _record("1", "2", "disulf3", pdbx_ptnr1_label_alt_id="A"),
    ]

    computation = _compute(tmp_path, atoms, records)

    assert len(computation.table) == 1
    assert computation.warnings == ()
    assert computation.status == "ok"


def test_a_record_naming_one_cysteine_twice_is_unmatched(tmp_path: Path) -> None:
    computation = _compute(tmp_path, _cysteine("1"), [_record("1", "1", "disulf9")])

    assert computation.table.empty
    assert computation.unmatched_record_count == 1
    assert computation.status == "degraded"
    assert computation.warnings == (
        "_struct_conn record 'disulf9' names the same cysteine twice, so it gives "
        "no row.",
    )


def test_label_fields_name_a_partner_only_without_author_fields(
    tmp_path: Path,
) -> None:
    atoms = [
        *_cysteine("101", label_seq="1"),
        *_cysteine("102", x=3.5, label_seq="2"),
    ]
    by_label = _record(
        "?",
        "?",
        "disulf1",
        ptnr1_auth_asym_id=None,
        ptnr2_auth_asym_id=None,
        ptnr1_label_asym_id="A",
        ptnr1_label_seq_id="1",
        ptnr2_label_asym_id="A",
        ptnr2_label_seq_id="2",
    )
    # Author fields that name no residue are not retried by label.
    by_wrong_author = {
        **by_label,
        "id": "disulf2",
        "ptnr1_auth_asym_id": "A",
        "ptnr1_auth_seq_id": "1",
        "ptnr2_auth_asym_id": "A",
        "ptnr2_auth_seq_id": "2",
    }

    computation = _compute(tmp_path, atoms, [by_label, by_wrong_author])

    assert computation.table["site_id"].tolist() == ["1ABC:1:A:101::SG:"]
    assert computation.table["detection_method"].tolist() == ["struct_conn"]
    assert computation.unmatched_record_count == 1
    assert "'disulf2'" in computation.warnings[0]


def test_a_free_cysteine_ligand_is_named_by_its_author_fields(tmp_path: Path) -> None:
    atoms = [
        *_cysteine("1"),
        *_cysteine("301", x=2.03, chain="B", label_seq="."),
    ]
    record = _record("1", "301", ptnr2_auth_asym_id="B", ptnr2_label_seq_id=None)

    computation = _compute(tmp_path, atoms, [record])

    row = computation.table.iloc[0]
    assert (row["partner_site_id"], row["detection_method"]) == (
        "1ABC:1:B:301::SG:",
        "struct_conn_and_distance",
    )


@pytest.mark.parametrize(
    ("occupancies", "total"),
    [(("0.33", "0.33", "0.34"), 1.0), (("0.33", "0.33", "0.33"), 0.99)],
)
def test_occupancy_sums_are_rounded_not_clamped(
    tmp_path: Path, occupancies: tuple[str, str, str], total: float
) -> None:
    positions = [(2.0, 0.0, 0.0), (0.0, 2.0, 0.0), (0.0, 0.0, 2.0)]
    atoms = [
        *_cysteine("1"),
        _atom("2", "CB", x=-1.0),
        *(
            _atom("2", x=x, y=y, z=z, altloc=altloc, occupancy=occupancy)
            for (x, y, z), altloc, occupancy in zip(
                positions, "ABC", occupancies, strict=True
            )
        ),
    ]

    row = _compute(tmp_path, atoms).table.iloc[0]

    assert row["sg_total_occupancy_min"] == total
    assert row["sg_conformer_occupancy_min"] == 0.33
    assert row["ss_distance_min_A"] == row["ss_distance_max_A"] == 2.0


def test_a_lone_labelled_sg_has_no_alternate_conformers(tmp_path: Path) -> None:
    atoms = [
        *_cysteine("1", altloc="A", occupancy="0.60"),
        *_cysteine("2", x=2.04),
    ]

    row = _compute(tmp_path, atoms).table.iloc[0]

    assert not row["sg_has_alternate_conformers"]
    assert row["sg_total_occupancy_min"] == 0.6


def test_an_unlabelled_sg_beside_one_labelled_sg_is_not_an_alternate(
    tmp_path: Path,
) -> None:
    atoms = [
        _atom("1", "CB", z=-1.8),
        _atom("1", occupancy="0.50"),
        _atom("1", z=0.5, altloc="A", occupancy="0.50"),
        *_cysteine("2", x=2.04),
    ]

    row = _compute(tmp_path, atoms).table.iloc[0]

    assert not row["sg_has_alternate_conformers"]
    assert row["sg_total_occupancy_min"] == 1.0


def test_the_kept_conformer_represents_a_residue_over_a_fuller_sg(
    tmp_path: Path,
) -> None:
    # Conformer A has the higher mean occupancy, (0.8 + 0.4) / 2 = 0.6 against
    # (0.2 + 0.6) / 2 = 0.4, so the atom table keeps it, although SG B has the
    # higher occupancy of the two SGs.
    atoms = [
        _atom("1", "CB", z=-1.8, altloc="A", occupancy="0.80"),
        _atom("1", altloc="A", occupancy="0.40"),
        _atom("1", "CB", y=2.5, z=-1.8, altloc="B", occupancy="0.20"),
        _atom("1", y=2.5, altloc="B", occupancy="0.60"),
        *_cysteine("2", x=2.04),
    ]

    computation = _compute(tmp_path, atoms)

    row = computation.table.iloc[0]
    assert (row["site_id"], row["pair_status"]) == ("1ABC:1:A:1::SG:A", "ok")
    assert "1ABC:1:A:1::SG:A" in set(computation.link.dropna())


@pytest.mark.parametrize(
    ("second_residue", "omit", "status"),
    [
        ([_atom("2", x=2.04)], (), "cb_missing"),
        (
            [_atom("2", "CB", x=2.04, z=-1.8, b_factor="0.00"), _atom("2", x=2.04)],
            (),
            "b_factor_unavailable",
        ),
        (
            [_atom("2", "CB", x=2.04, z=-1.8), _atom("2", x=2.04, b_factor="?")],
            (),
            "b_factor_unavailable",
        ),
        (_cysteine("2", x=2.04), ("B_iso_or_equiv",), "b_factor_unavailable"),
        (
            [
                _atom("2", "CB", x=2.04, z=-1.8, altloc="B", occupancy="0.50"),
                _atom("2", x=2.04, altloc="A", occupancy="0.50"),
            ],
            (),
            "cb_missing",
        ),
    ],
)
def test_one_residue_without_a_ratio_leaves_both_ratios_missing(
    tmp_path: Path,
    second_residue: list[dict[str, str]],
    omit: tuple[str, ...],
    status: str,
) -> None:
    computation = _compute(tmp_path, [*_cysteine("1"), *second_residue], omit=omit)

    row = computation.table.iloc[0]
    assert row["pair_status"] == status
    assert math.isnan(row["sg_cb_b_factor_ratio_min"])
    assert math.isnan(row["sg_cb_b_factor_ratio_max"])


def test_both_ratios_are_reported_when_each_residue_has_one(tmp_path: Path) -> None:
    computation = _compute(
        tmp_path, [*_cysteine("1"), *_cysteine("2", x=2.04, carbon_b_factor="10")]
    )

    row = computation.table.iloc[0]
    assert row["pair_status"] == "ok"
    # 20 / 18 for the first residue and 20 / 10 for the second.
    assert row["sg_cb_b_factor_ratio_min"] == pytest.approx(20 / 18)
    assert row["sg_cb_b_factor_ratio_max"] == pytest.approx(2.0)


def test_an_unlabelled_cb_represents_a_split_sg(tmp_path: Path) -> None:
    atoms = [
        *_cysteine("1"),
        _atom("2", "CB", x=2.04, z=-1.8, b_factor="16"),
        _atom("2", x=2.04, altloc="A", occupancy="0.70", b_factor="24"),
        _atom("2", y=2.04, altloc="B", occupancy="0.30", b_factor="40"),
    ]

    row = _compute(tmp_path, atoms).table.iloc[0]

    assert row["sg_cb_b_factor_ratio_max"] == pytest.approx(24 / 16)
    assert row["pair_status"] == "ok"


@pytest.mark.parametrize(
    ("first_residue", "second_residue", "status"),
    [
        # A missing occupancy outranks a missing CB.
        (
            _cysteine("1", occupancy="?"),
            [_atom("2", x=2.04)],
            "occupancy_missing",
        ),
        # A missing CB outranks a placeholder B-factor.
        (
            _cysteine("1", sulfur_b_factor="0.00"),
            [_atom("2", x=2.04)],
            "cb_missing",
        ),
        # A residue without a usable SG outranks everything.
        (
            _cysteine("1", occupancy="0.00", sulfur_b_factor="0.00"),
            [_atom("2", x=2.04, occupancy="?")],
            "no_usable_sg",
        ),
    ],
)
def test_the_pair_status_names_the_first_gap(
    tmp_path: Path,
    first_residue: list[dict[str, str]],
    second_residue: list[dict[str, str]],
    status: str,
) -> None:
    computation = _compute(
        tmp_path, [*first_residue, *second_residue], [_record("1", "2")]
    )

    assert computation.table["pair_status"].tolist() == [status]


@pytest.mark.parametrize(
    ("first", "second", "expected_first"),
    [
        (_cysteine("10"), _cysteine("9", x=2.04), "1ABC:1:A:9::SG:"),
        (
            _cysteine("52", ins_code="A", label_seq="53"),
            _cysteine("52", x=2.04),
            "1ABC:1:A:52::SG:",
        ),
        (
            _cysteine("1", chain="B"),
            _cysteine("5", x=2.04),
            "1ABC:1:A:5::SG:",
        ),
    ],
)
def test_the_residue_that_sorts_first_leads_the_row(
    tmp_path: Path,
    first: list[dict[str, str]],
    second: list[dict[str, str]],
    expected_first: str,
) -> None:
    computation = _compute(tmp_path, [*first, *second])

    assert computation.table["site_id"].tolist() == [expected_first]


def test_file_order_does_not_change_the_table(tmp_path: Path) -> None:
    atoms = [
        *_cysteine("9"),
        *_cysteine("10", x=2.04),
        *_cysteine("52", x=20.0),
        *_cysteine("52", x=22.7, ins_code="A", label_seq="53"),
        *_cysteine("3", x=40.0, chain="B"),
        *_cysteine("4", x=43.6, chain="B"),
        *_cysteine("9", x=5.0, model="2"),
        *_cysteine("10", x=7.04, model="2"),
    ]
    records = [_record("3", "4", ptnr1_auth_asym_id="B", ptnr2_auth_asym_id="B")]
    reversed_records = [
        _record("4", "3", ptnr1_auth_asym_id="B", ptnr2_auth_asym_id="B")
    ]

    in_order = _compute(tmp_path, atoms, records).table
    shuffled = _compute(tmp_path, atoms[::-1], reversed_records).table

    pd.testing.assert_frame_equal(in_order, shuffled)
    assert in_order["site_id"].tolist() == [
        "1ABC:1:A:9::SG:",
        "1ABC:1:A:52::SG:",
        "1ABC:1:B:3::SG:",
        "1ABC:2:A:9::SG:",
    ]


def test_the_link_holds_each_paired_sg_own_site_id_on_the_atom_index(
    tmp_path: Path,
) -> None:
    atom_table, mmcif = _structure(
        tmp_path,
        [*_cysteine("1"), *_cysteine("2", x=2.04), *_cysteine("3", y=2.5)],
    )
    atom_table.index = atom_table.index + 100

    computation = compute_structure_disulfides(atom_table, mmcif, pdb_id="1abc")

    # SG 1 is within 3.0 Å of SG 2 and of SG 3, so it is in two rows and still
    # links by its own site_id.
    assert len(computation.table) == 2
    assert computation.link.index.equals(atom_table.index)
    assert computation.link.dropna().to_dict() == {
        101: "1ABC:1:A:1::SG:",
        103: "1ABC:1:A:2::SG:",
        105: "1ABC:1:A:3::SG:",
    }


def test_a_structure_without_cysteine_gives_an_empty_table(minimal_cif: Path) -> None:
    atoms, mmcif = _read(minimal_cif)

    computation = compute_structure_disulfides(atoms, mmcif, pdb_id="1abc")

    pd.testing.assert_frame_equal(computation.table, DISULFIDES_TABLE.empty_dataframe())
    assert computation.link.isna().all()
    assert computation.status == "ok"


def test_two_atoms_with_one_site_id_are_refused(tmp_path: Path) -> None:
    atoms = [*_cysteine("1"), _atom("1"), *_cysteine("2", x=2.04)]

    with pytest.raises(ValueError, match="share the site identifier 1ABC:1:A:1::SG:"):
        _compute(tmp_path, atoms)


def test_an_atom_table_that_names_an_atom_otherwise_is_refused(
    tmp_path: Path,
) -> None:
    atom_table, mmcif = _structure(tmp_path, [*_cysteine("1"), *_cysteine("2", x=2.04)])
    renamed = atom_table.copy()
    renamed.loc[1, "chain_id"] = "Z"

    with pytest.raises(RuntimeError, match="does not match the atom table"):
        compute_structure_disulfides(renamed, mmcif, pdb_id="1abc")


def test_an_atom_table_with_an_atom_the_file_lacks_is_refused(
    tmp_path: Path,
) -> None:
    atom_table, mmcif = _structure(tmp_path, [*_cysteine("1"), *_cysteine("2", x=2.04)])
    extra = pd.concat(
        [atom_table, atom_table.iloc[[1]].assign(**{"_atom_site.id": "99"})]
    )

    with pytest.raises(RuntimeError, match="holds no atom for"):
        compute_structure_disulfides(extra.reset_index(drop=True), mmcif, pdb_id="1abc")


def test_the_fixture_rows_pass_the_table_checks(disulfide_cif: Path) -> None:
    _atoms, computation = _fixture(disulfide_cif)
    table = computation.table

    validated = validate_tables(
        "disulfides", (DISULFIDES_TABLE,), {"disulfides": table}
    )

    pd.testing.assert_frame_equal(validated["disulfides"], table)
    broken = table.assign(pair_status="broken")
    with pytest.raises(ValueError, match="undeclared categories"):
        validate_tables("disulfides", (DISULFIDES_TABLE,), {"disulfides": broken})


def test_numbers_with_a_standard_uncertainty_are_read_as_the_parser_reads_them(
    tmp_path: Path,
) -> None:
    atoms = [
        *_cysteine("1"),
        _atom("2", "CB", x="2.040(3)", z=-1.8, b_factor="18(1)"),
        _atom("2", x="2.040(3)", occupancy="1.00(2)", b_factor="20(1)"),
    ]

    computation = _compute(tmp_path, atoms, [_record("1", "2")])

    row = computation.table.iloc[0]
    assert row["detection_method"] == "struct_conn_and_distance"
    assert row["ss_distance_min_A"] == pytest.approx(2.04, abs=1e-9)
    assert row["sg_total_occupancy_min"] == 1.0
    assert row["sg_cb_b_factor_ratio_max"] == pytest.approx(20 / 18)
    assert row["pair_status"] == "ok"


def test_a_record_to_a_cysteine_without_side_chain_says_so(tmp_path: Path) -> None:
    atoms = [*_cysteine("1"), _atom("2", "CA", x=3.5)]

    computation = _compute(tmp_path, atoms, [_record("1", "2", "disulf1")])

    assert computation.warnings == (
        "_struct_conn record 'disulf1' names a cysteine with no SG or CB atom in "
        "the structure, so it gives no row.",
    )


def test_cysteines_with_a_cb_but_no_sg_are_counted_once_per_model(
    tmp_path: Path,
) -> None:
    atoms = [
        *_cysteine("1"),
        _atom("2", "CB", x=3.5),
        _atom("3", "CB", x=7.0),
        _atom("3", "CB", x=7.0, altloc="B", occupancy="0.40"),
        _atom("2", "CB", x=3.5, model="2"),
    ]

    computation = _compute(tmp_path, atoms)

    assert computation.table.empty
    assert computation.cysteine_without_sg_count == 3
    assert computation.status == "ok"


@pytest.mark.parametrize(
    ("residue", "deficit"),
    [
        # Occupancies rounded to two decimals sum to 0.98 on SG and CB alike.
        (
            [
                *_cysteine("2", x=2.04, altloc="A", occupancy="0.11"),
                *_cysteine("2", x=2.04, y=0.1, altloc="B", occupancy="0.36"),
                *_cysteine("2", x=2.04, y=0.2, altloc="C", occupancy="0.51"),
            ],
            0.0,
        ),
        # A whole residue at partial occupancy.
        (_cysteine("2", x=2.04, occupancy="0.50"), 0.0),
        # Part of the SG is not modelled.
        (
            [
                _atom("2", "CB", x=2.04, z=-1.8),
                _atom("2", x=2.04, occupancy="0.75"),
            ],
            0.25,
        ),
    ],
)
def test_the_occupancy_deficit_measures_the_sg_against_its_own_cb(
    tmp_path: Path, residue: list[dict[str, str]], deficit: float
) -> None:
    row = _compute(tmp_path, [*_cysteine("1"), *residue]).table.iloc[0]

    assert row["sg_occupancy_deficit_max"] == deficit


def test_a_missing_cb_occupancy_leaves_the_deficit_missing(tmp_path: Path) -> None:
    atoms = [
        *_cysteine("1"),
        _atom("2", "CB", x=2.04, z=-1.8, occupancy="?"),
        _atom("2", x=2.04),
    ]

    row = _compute(tmp_path, atoms).table.iloc[0]

    assert math.isnan(row["sg_occupancy_deficit_max"])
    assert row["sg_total_occupancy_min"] == 1.0
    assert row["pair_status"] == "occupancy_missing"


def test_a_residue_without_a_cb_leaves_the_deficit_missing(tmp_path: Path) -> None:
    row = _compute(tmp_path, [*_cysteine("1"), _atom("2", x=2.04)]).table.iloc[0]

    assert math.isnan(row["sg_occupancy_deficit_max"])
    assert row["pair_status"] == "cb_missing"


def test_a_repeated_warning_is_given_once(tmp_path: Path) -> None:
    records = [_record("1", "9", "disulf1"), _record("1", "9", "disulf1")]

    computation = _compute(tmp_path, _cysteine("1"), records)

    assert computation.unmatched_record_count == 2
    assert len(computation.warnings) == 1
