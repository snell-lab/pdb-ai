from __future__ import annotations

from collections.abc import Mapping

import pytest

from pdbai.engines.analyses.disulfides.records import (
    DisulfideRecords,
    read_disulfide_records,
)
from pdbai.engines.analyses.struct_conn import PartnerAddress, StructConnRecord


def _record(
    record_id: str = "disulf1", /, **items: str | None
) -> dict[str, str | None]:
    """Return one _struct_conn row joining the SG atoms of Cys A 10 and A 20."""
    return {
        "id": record_id,
        "conn_type_id": "disulf",
        "ptnr1_auth_asym_id": "A",
        "ptnr1_auth_seq_id": "10",
        "ptnr1_label_comp_id": "CYS",
        "ptnr1_label_atom_id": "SG",
        "ptnr2_auth_asym_id": "A",
        "ptnr2_auth_seq_id": "20",
        "ptnr2_label_comp_id": "CYS",
        "ptnr2_label_atom_id": "SG",
        **items,
    }


def _flat(*rows: Mapping[str, str | None]) -> dict[str, list[str | None]]:
    """Return a flat mmCIF dictionary whose _struct_conn holds ``rows``."""
    items = dict.fromkeys(item for row in rows for item in row)
    return {f"_struct_conn.{item}": [row.get(item) for row in rows] for item in items}


def test_a_disulf_record_keeps_both_partners_author_and_label_fields() -> None:
    records = read_disulfide_records(
        _flat(
            _record(
                ptnr1_label_asym_id="A",
                ptnr1_label_seq_id="1",
                pdbx_ptnr1_PDB_ins_code="B",
                ptnr2_auth_asym_id=None,
                ptnr2_auth_seq_id="?",
                ptnr2_label_asym_id="C",
                ptnr2_label_seq_id=".",
            )
        )
    )

    assert records == DisulfideRecords(
        records=(
            StructConnRecord(
                "disulf1",
                "disulf",
                PartnerAddress(
                    label_asym_id="A",
                    label_seq_id="1",
                    auth_asym_id="A",
                    auth_seq_id="10",
                    ins_code="B",
                    component="CYS",
                    atom_name="SG",
                ),
                PartnerAddress(
                    label_asym_id="C",
                    label_seq_id="",
                    auth_asym_id="",
                    auth_seq_id="",
                    ins_code="",
                    component="CYS",
                    atom_name="SG",
                ),
                same_operator=True,
            ),
        ),
        symmetry_record_count=0,
    )
    assert records.records[0].first.has_author_fields
    assert not records.records[0].second.has_author_fields


def test_the_connection_type_is_matched_in_any_case() -> None:
    records = read_disulfide_records(_flat(_record(conn_type_id=" DISULF ")))

    assert [record.record_id for record in records.records] == ["disulf1"]


@pytest.mark.parametrize("connection_type", ["covale", "metalc", None, "?", "."])
def test_a_row_of_another_or_no_connection_type_gives_no_record(
    connection_type: str | None,
) -> None:
    records = read_disulfide_records(_flat(_record(conn_type_id=connection_type)))

    assert records == DisulfideRecords((), 0)


def test_a_file_without_struct_conn_or_its_type_gives_no_record() -> None:
    assert read_disulfide_records({}) == DisulfideRecords((), 0)
    without_type = _record()
    del without_type["conn_type_id"]
    assert read_disulfide_records(_flat(without_type)) == DisulfideRecords((), 0)


def test_partners_under_different_operators_are_only_counted() -> None:
    records = read_disulfide_records(
        _flat(
            _record("disulf1", ptnr2_symmetry="2_655"),
            _record("disulf2", ptnr1_symmetry="1_555", ptnr2_symmetry="1_555"),
        )
    )

    assert [record.record_id for record in records.records] == ["disulf2"]
    assert records.symmetry_record_count == 1


@pytest.mark.parametrize(
    ("first_operator", "second_operator"),
    [
        ("2_555", "2_555"),
        (None, None),
        (None, "1_555"),
        ("1555", "1_555"),
        ("?", " 1_555 "),
    ],
)
def test_partners_under_one_operator_are_within_one_molecule(
    first_operator: str | None, second_operator: str | None
) -> None:
    records = read_disulfide_records(
        _flat(_record(ptnr1_symmetry=first_operator, ptnr2_symmetry=second_operator))
    )

    assert len(records.records) == 1
    assert records.symmetry_record_count == 0


def test_a_residue_joined_to_its_own_symmetry_mate_is_counted() -> None:
    # A disulfide across a two-fold axis names one residue twice, so the
    # operators decide before any check that the residues differ.
    records = read_disulfide_records(
        _flat(_record(ptnr2_auth_seq_id="10", ptnr2_symmetry="2_555"))
    )

    assert records == DisulfideRecords((), 1)


@pytest.mark.parametrize(
    "items",
    [
        {"ptnr2_label_comp_id": "BME"},
        {"ptnr2_label_comp_id": "GSH"},
        {"ptnr1_label_comp_id": "CSO"},
        {"ptnr2_auth_comp_id": "DCY", "ptnr2_label_comp_id": "CYS"},
        {"ptnr2_label_comp_id": None, "ptnr2_auth_comp_id": "BME"},
        {"ptnr2_label_atom_id": "N"},
        {"ptnr2_label_atom_id": None, "ptnr2_auth_atom_id": "CB"},
    ],
)
def test_a_record_to_another_residue_or_atom_is_skipped_without_a_count(
    items: dict[str, str | None],
) -> None:
    # A symmetry operator that would otherwise be counted shows the record
    # was dropped before that check.
    records = read_disulfide_records(_flat(_record(ptnr2_symmetry="2_655", **items)))

    assert records == DisulfideRecords((), 0)


def test_missing_component_and_atom_names_do_not_put_a_record_out_of_scope() -> None:
    records = read_disulfide_records(
        _flat(
            _record(
                ptnr1_label_comp_id=None,
                ptnr1_label_atom_id="?",
                ptnr2_label_comp_id="cys",
                ptnr2_auth_comp_id=None,
            )
        )
    )

    assert len(records.records) == 1
