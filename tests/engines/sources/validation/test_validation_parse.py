from __future__ import annotations

import xml.etree.ElementTree as ET
from typing import cast

import pytest

from pdbai.engines.sources.validation.parse import parse_validation_xml


def _report(*subgroups: str, entry: str = '<Entry clashscore="2.5"/>') -> bytes:
    return (
        '<?xml version="1.0"?>\n<wwPDB-validation-information>\n'
        + entry
        + "\n"
        + "\n".join(subgroups)
        + "\n</wwPDB-validation-information>\n"
    ).encode()


def _subgroup(resnum: str, altcode: str, rscc: str, *, rota: str = "m") -> str:
    return (
        f'<ModelledSubgroup model="1" chain="A" resnum="{resnum}" icode=" " '
        f'altcode="{altcode}" resname="SER" rscc="{rscc}" rsr="0.1" '
        f'rama="Favored" rota="{rota}"/>'
    )


def _residues(report: bytes) -> dict[str, dict[str, object]]:
    return cast(dict[str, dict[str, object]], parse_validation_xml(report)["residues"])


def test_a_report_gives_its_entry_metrics_and_residue_values() -> None:
    # The attributes as a real report writes them, with a blank altcode and
    # insertion code, and a residue that has no Ramachandran class.
    report = _report(
        '<ModelledSubgroup NatomsEDS="7" altcode=" " avgoccu="1.000" chain="A" '
        'ent="1" icode=" " model="1" owab="28.790" resname="PRO" resnum="1" '
        'rota="Cg_endo" rscc="0.908" rsr="0.143" rsrz="0.706" said="A" seq="1"/>',
        entry=(
            '<Entry clashscore="4.01" percent-rama-outliers="0.00" '
            'percent-rota-outliers="2.44"/>'
        ),
    )

    parsed = parse_validation_xml(report)

    assert parsed == {
        "residues": {
            "1:A:1": {
                "rscc": 0.908,
                "rsr": 0.143,
                "rama_class": None,
                "rotamer_class": "Cg_endo",
            }
        },
        "clashscore": 4.01,
        "percent_rama_outliers": 0.0,
        "percent_rotamer_outliers": 2.44,
    }


def test_alternate_conformer_a_wins_over_one_listed_after_it() -> None:
    report = _report(_subgroup("5", "A", "0.91"), _subgroup("5", "B", "0.72"))

    residues = _residues(report)

    assert sorted(residues) == ["1:A:5", "1:A:5:A", "1:A:5:B"]
    assert residues["1:A:5"]["rscc"] == 0.91


def test_alternate_conformer_a_wins_over_one_listed_before_it() -> None:
    report = _report(_subgroup("5", "B", "0.72"), _subgroup("5", "A", "0.91"))

    residues = _residues(report)

    assert residues["1:A:5"]["rscc"] == 0.91


def test_first_listed_conformer_wins_when_none_is_blank_or_a() -> None:
    report = _report(_subgroup("5", "C", "0.64"), _subgroup("5", "B", "0.72"))

    residues = _residues(report)

    assert residues["1:A:5"]["rscc"] == 0.64


def test_each_conformer_is_kept_under_its_own_key() -> None:
    report = _report(
        _subgroup("5", "A", "0.91", rota="mt"),
        _subgroup("5", "B", "0.72", rota="OUTLIER"),
        _subgroup("6", " ", "0.88"),
    )

    residues = _residues(report)

    assert residues["1:A:5:A"]["rotamer_class"] == "mt"
    assert residues["1:A:5:B"]["rotamer_class"] == "OUTLIER"
    assert residues["1:A:5:B"]["rscc"] == 0.72
    # A residue without alternates has no conformer key.
    assert "1:A:6" in residues
    assert not any(key.startswith("1:A:6:") for key in residues)


def test_an_insertion_code_is_appended_to_the_residue_number() -> None:
    report = _report(
        '<ModelledSubgroup model="2" chain="B" resnum="-3" icode="A" rscc="0.5"/>'
    )

    assert list(_residues(report)) == ["2:B:-3A"]


def test_a_subgroup_without_a_model_belongs_to_model_one() -> None:
    report = _report('<ModelledSubgroup chain="A" resnum="4" rscc="0.5"/>')

    assert list(_residues(report)) == ["1:A:4"]


def test_a_subgroup_without_a_chain_or_residue_number_is_skipped() -> None:
    report = _report(
        '<ModelledSubgroup model="1" chain=" " resnum="4" rscc="0.5"/>',
        '<ModelledSubgroup model="1" chain="A" rscc="0.5"/>',
    )

    assert _residues(report) == {}


def test_a_malformed_number_or_a_blank_class_reads_as_missing() -> None:
    report = _report(
        '<ModelledSubgroup model="1" chain="A" resnum="4" rscc="n/a" rsr="nan" '
        'rama=" " rota=""/>',
        entry='<Entry clashscore="" percent-rama-outliers="1e2"/>',
    )

    parsed = parse_validation_xml(report)

    assert _residues(report)["1:A:4"] == {
        "rscc": None,
        "rsr": None,
        "rama_class": None,
        "rotamer_class": None,
    }
    assert parsed["clashscore"] is None
    assert parsed["percent_rama_outliers"] is None
    assert parsed["percent_rotamer_outliers"] is None


def test_a_report_without_an_entry_gives_residues_and_no_metrics() -> None:
    # The root element's attributes are not the entry's summary, so they are
    # not read as metrics.
    report = (
        b'<wwPDB-validation-information clashscore="9.9">'
        b'<ModelledSubgroup model="1" chain="A" resnum="4" rscc="0.5"/>'
        b"</wwPDB-validation-information>"
    )

    parsed = parse_validation_xml(report)

    assert _residues(report)["1:A:4"]["rscc"] == 0.5
    assert parsed["clashscore"] is None
    assert parsed["percent_rama_outliers"] is None
    assert parsed["percent_rotamer_outliers"] is None


def test_xml_that_is_not_a_validation_report_is_refused() -> None:
    with pytest.raises(ValueError, match="'Error' is not a validation report"):
        parse_validation_xml(b"<Error><Code>NoSuchKey</Code></Error>")


def test_bytes_that_are_not_xml_raise_a_parse_error() -> None:
    with pytest.raises(ET.ParseError):
        parse_validation_xml(b"<html")
