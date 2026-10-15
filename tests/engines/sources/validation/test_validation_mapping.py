from __future__ import annotations

import pandas as pd
import pytest

from pdbai.engines.core import EngineContext, EngineOptions
from pdbai.engines.sources.validation.artifacts import VALIDATION_REPORT
from pdbai.engines.sources.validation.engine import VALIDATION_ANNOTATION_ENGINE
from pdbai.engines.sources.validation.fetch import (
    VALIDATION_STATUS_AVAILABLE,
    ValidationReportResult,
)


def _context(
    pdb_id: str,
    atom_dataframe: pd.DataFrame,
    options: EngineOptions,
    *,
    report: dict[str, object] | None = None,
) -> EngineContext:
    # The report stands in for the one the acquisition engine publishes. With
    # no report, or an empty one, nothing is published. That is how a
    # structure without a validation report looks.
    context = EngineContext(pdb_id, atom_dataframe, options)
    if report:
        context.artifacts.put(
            VALIDATION_REPORT,
            ValidationReportResult(report=report, status=VALIDATION_STATUS_AVAILABLE),
        )
    return context


_COLUMNS = (
    "eds_rscc",
    "eds_rsr",
    "rama_class",
    "rotamer_class",
    "clashscore",
    "percent_rama_outliers",
    "percent_rotamer_outliers",
)


def _identity(res_num: str, atom_name: str, alt_id: str = ".") -> dict[str, str]:
    return {
        "pdb_id": "1abc",
        "model_num": "1",
        "chain_id": "A",
        "_atom_site.auth_asym_id": "A",
        "res_num": res_num,
        "_atom_site.auth_seq_id": res_num,
        "_atom_site.pdbx_PDB_ins_code": "",
        "_atom_site.label_alt_id": alt_id,
        "res_name": "GLY",
        "atom_name": atom_name,
    }


def test_validation_maps_residue_values_onto_every_atom_of_the_residue() -> None:
    atom_dataframe = pd.DataFrame(
        [
            _identity("7", "N"),
            _identity("7", "CA"),
            _identity("8", "CA"),
            _identity("9", "CA"),
        ]
    )
    context = _context(
        "1ABC",
        atom_dataframe,
        EngineOptions(values={"include_validation": True}),
        report={
            "residues": {
                "1:A:7": {
                    "rscc": 0.95,
                    "rsr": 0.11,
                    "rama_class": "Favored",
                    "rotamer_class": "OUTLIER",
                },
                "1:A:8": {"rscc": 0.5, "rsr": 0.4},
            },
            "clashscore": 3.5,
            "percent_rama_outliers": 0.25,
            "percent_rotamer_outliers": 1.5,
        },
    )

    columns = VALIDATION_ANNOTATION_ENGINE.run(context).columns
    assert columns is not None

    assert columns["eds_rscc"].tolist()[:3] == [0.95, 0.95, 0.5]
    assert columns["eds_rsr"].tolist()[:3] == [0.11, 0.11, 0.4]
    assert columns[["eds_rscc", "eds_rsr"]].iloc[3].isna().all()
    assert columns["rama_class"].tolist() == ["Favored", "Favored", None, None]
    assert columns["rotamer_class"].tolist() == ["OUTLIER", "OUTLIER", None, None]
    assert columns["clashscore"].tolist() == [3.5] * 4


def test_validation_columns_are_present_without_any_report() -> None:
    atom_dataframe = pd.DataFrame([_identity("7", "CA")])
    context = _context(
        "1ABC", atom_dataframe, EngineOptions(values={"include_validation": True})
    )

    columns = VALIDATION_ANNOTATION_ENGINE.run(context).columns
    assert columns is not None

    for name in _COLUMNS:
        assert name in columns.columns
        assert columns[name].isna().all(), name


@pytest.mark.parametrize("model_column", ["model_num", "_atom_site.pdbx_PDB_model_num"])
def test_validation_keys_are_model_aware(model_column: str) -> None:
    atom_dataframe = pd.DataFrame([_identity("7", "CA"), _identity("7", "CA")])
    atom_dataframe[model_column] = ["1", "2"]
    if model_column != "model_num":
        atom_dataframe = atom_dataframe.drop(columns=["model_num"])

    context = _context(
        "1ABC",
        atom_dataframe,
        EngineOptions(values={"include_validation": True}),
        report={"residues": {"1:A:7": {"rscc": 0.9}}},
    )

    columns = VALIDATION_ANNOTATION_ENGINE.run(context).columns
    assert columns is not None

    assert columns["eds_rscc"].iloc[0] == 0.9
    assert pd.isna(columns["eds_rscc"].iloc[1])


def _conformer_report() -> dict[str, object]:
    # Residue 7 has conformers A and B. Its residue key holds A's values, as
    # parse_validation_xml keeps them.
    conformer_a = {"rscc": 0.9, "rsr": 0.1, "rama_class": "Favored"}
    conformer_b = {"rscc": 0.6, "rsr": 0.3, "rama_class": "OUTLIER"}
    return {
        "residues": {
            "1:A:7": conformer_a,
            "1:A:7:A": conformer_a,
            "1:A:7:B": conformer_b,
        }
    }


def test_a_residue_takes_the_values_of_the_conformer_the_parser_kept() -> None:
    # The parser kept conformer B. The backbone atom without an altcode is
    # shared by both conformers and takes B's values with the rest.
    atom_dataframe = pd.DataFrame(
        [_identity("7", "N"), _identity("7", "CB", "B"), _identity("7", "OG", "B")]
    )
    context = _context(
        "1ABC",
        atom_dataframe,
        EngineOptions(values={"include_validation": True}),
        report=_conformer_report(),
    )

    columns = VALIDATION_ANNOTATION_ENGINE.run(context).columns
    assert columns is not None

    assert columns["eds_rscc"].tolist() == [0.6, 0.6, 0.6]
    assert columns["rama_class"].tolist() == ["OUTLIER"] * 3


def test_a_conformer_the_report_does_not_list_takes_the_residue_values() -> None:
    atom_dataframe = pd.DataFrame([_identity("7", "N"), _identity("7", "CB", "C")])
    context = _context(
        "1ABC",
        atom_dataframe,
        EngineOptions(values={"include_validation": True}),
        report=_conformer_report(),
    )

    columns = VALIDATION_ANNOTATION_ENGINE.run(context).columns
    assert columns is not None

    assert columns["eds_rscc"].tolist() == [0.9, 0.9]


def test_validation_columns_have_their_declared_types() -> None:
    # A cached report is read back from JSON, so its values are checked
    # again. A value that is not a number, or a class that is not text, is
    # missing. A number written as decimal text is still read.
    atom_dataframe = pd.DataFrame(
        [_identity("7", "CA"), _identity("8", "CA"), _identity("9", "CA")]
    )
    context = _context(
        "1ABC",
        atom_dataframe,
        EngineOptions(values={"include_validation": True}),
        report={
            "residues": {
                "1:A:7": {"rscc": 0.9, "rsr": 0.2, "rama_class": "Favored"},
                "1:A:8": {"rscc": "high", "rsr": True, "rama_class": 3},
                "1:A:9": "not a residue",
            },
            "clashscore": "4.5",
            "percent_rama_outliers": None,
        },
    )

    columns = VALIDATION_ANNOTATION_ENGINE.run(context).columns
    assert columns is not None

    for name in ("eds_rscc", "eds_rsr", *_COLUMNS[4:]):
        assert columns[name].dtype == "float64", name
    assert columns["eds_rscc"].iloc[0] == 0.9
    assert columns[["eds_rscc", "eds_rsr"]].iloc[1:].isna().all().all()
    assert columns["rama_class"].tolist() == ["Favored", None, None]
    assert columns["clashscore"].tolist() == [4.5] * 3
    assert columns["percent_rama_outliers"].isna().all()


def test_validation_columns_are_empty_for_a_table_without_atoms() -> None:
    context = _context(
        "1ABC",
        pd.DataFrame(),
        EngineOptions(values={"include_validation": True}),
        report={"residues": {"1:A:7": {"rscc": 0.9}}, "clashscore": 1.0},
    )

    columns = VALIDATION_ANNOTATION_ENGINE.run(context).columns
    assert columns is not None

    assert list(columns.columns) == list(_COLUMNS)
    assert columns.empty
