from __future__ import annotations

import pytest

from pdbai.engines.sources.alphafold.model_fetch import parse_afdb_model_cif

_METRICS = """loop_
_ma_qa_metric.id
_ma_qa_metric.name
_ma_qa_metric.type
_ma_qa_metric.description
1 pLDDT pLDDT 'predicted lddt'
2 PAE PAE 'predicted aligned error'
"""

# Metric 2 is PAE, so its rows must never be read as pLDDT.
_LOCAL = """loop_
_ma_qa_metric_local.label_seq_id
_ma_qa_metric_local.metric_id
_ma_qa_metric_local.metric_value
1 1 90.0
2 1 80.0
1 2 5.0
2 2 6.0
"""

# AlphaFold writes each residue's pLDDT into the B-factor of its atoms, and
# only the CA atom's is read. The N row is a decoy that a parser reading
# every atom would pick up.
_CA_BFACTORS = """loop_
_atom_site.label_atom_id
_atom_site.label_seq_id
_atom_site.B_iso_or_equiv
CA 1 70.0
N  1 99.9
CA 2 60.0
"""


def _document(*blocks: str) -> str:
    return "data_model\n" + "".join(blocks)


def _scores(cif_text: str) -> dict[str, float]:
    return parse_afdb_model_cif(cif_text)[0]


def test_local_quality_metric_is_preferred() -> None:
    assert _scores(_document(_METRICS, _LOCAL, _CA_BFACTORS)) == {
        "1": 90.0,
        "2": 80.0,
    }


def test_only_the_metric_declared_as_plddt_is_read() -> None:

    scores = _scores(_document(_METRICS, _LOCAL))

    assert scores == {"1": 90.0, "2": 80.0}
    assert 5.0 not in scores.values()


def test_ca_b_factors_are_the_fallback_for_older_files() -> None:

    scores = _scores(_document(_CA_BFACTORS))

    assert scores == {"1": 70.0, "2": 60.0}, "non-CA atoms must be ignored"


def test_fallback_is_used_when_no_declared_metric_is_plddt() -> None:
    metrics = _METRICS.replace("1 pLDDT pLDDT 'predicted lddt'", "1 PAE PAE 'error'")

    assert _scores(_document(metrics, _LOCAL, _CA_BFACTORS)) == {
        "1": 70.0,
        "2": 60.0,
    }


def test_fallback_is_used_when_local_rows_carry_no_metric_id() -> None:

    local = """loop_
_ma_qa_metric_local.label_seq_id
_ma_qa_metric_local.metric_value
1 90.0
"""

    assert _scores(_document(_METRICS, local, _CA_BFACTORS)) == {
        "1": 70.0,
        "2": 60.0,
    }


def test_conflicting_duplicate_values_discard_the_whole_file() -> None:

    conflicting = """loop_
_ma_qa_metric_local.label_seq_id
_ma_qa_metric_local.metric_id
_ma_qa_metric_local.metric_value
1 1 90.0
1 1 40.0
"""

    assert _scores(_document(_METRICS, conflicting, _CA_BFACTORS)) == {}


def test_agreeing_duplicate_values_are_accepted() -> None:
    agreeing = """loop_
_ma_qa_metric_local.label_seq_id
_ma_qa_metric_local.metric_id
_ma_qa_metric_local.metric_value
1 1 90.0
1 1 90.0
"""

    assert _scores(_document(_METRICS, agreeing)) == {"1": 90.0}


@pytest.mark.parametrize("value", ["180.0", "-1.0", "nan", "inf", "not-a-number", "?"])
def test_values_outside_the_plddt_scale_are_dropped(value: str) -> None:

    local = f"""loop_
_ma_qa_metric_local.label_seq_id
_ma_qa_metric_local.metric_id
_ma_qa_metric_local.metric_value
1 1 {value}
2 1 80.0
"""

    assert _scores(_document(_METRICS, local)) == {"2": 80.0}


@pytest.mark.parametrize("boundary", ["0.0", "100.0"])
def test_scale_boundaries_are_kept(boundary: str) -> None:
    local = f"""loop_
_ma_qa_metric_local.label_seq_id
_ma_qa_metric_local.metric_id
_ma_qa_metric_local.metric_value
1 1 {boundary}
"""

    assert _scores(_document(_METRICS, local)) == {"1": float(boundary)}


@pytest.mark.parametrize(
    "text", ["", "not a cif at all", "data_empty\n", "<html>404</html>"]
)
def test_unreadable_documents_yield_no_scores(text: str) -> None:
    assert _scores(text) == {}


# The layout of an AlphaFold Database ModelCIF file, cut down to two
# residues. Metric 1 is the global pLDDT and metric 2 the local one.
_ALPHAFOLD_LAYOUT = """loop_
_ma_qa_metric.id
_ma_qa_metric.mode
_ma_qa_metric.name
_ma_qa_metric.software_group_id
_ma_qa_metric.type
1 global pLDDT 1 pLDDT
2 local  pLDDT 1 pLDDT
#
_ma_qa_metric_global.metric_id    1
_ma_qa_metric_global.metric_value 98.06
#
loop_
_ma_qa_metric_local.label_asym_id
_ma_qa_metric_local.label_comp_id
_ma_qa_metric_local.label_seq_id
_ma_qa_metric_local.metric_id
_ma_qa_metric_local.metric_value
A MET 1 2 65.38
A VAL 2 2 91.12
#
loop_
_atom_site.label_atom_id
_atom_site.label_comp_id
_atom_site.label_seq_id
_atom_site.B_iso_or_equiv
N  MET 1 65.38
CA MET 1 65.38
CA VAL 2 91.12
"""


def test_an_alphafold_model_gives_each_residue_its_score_and_name() -> None:
    assert parse_afdb_model_cif(_document(_ALPHAFOLD_LAYOUT)) == (
        {"1": 65.38, "2": 91.12},
        {"1": "MET", "2": "VAL"},
    )


def test_a_metric_category_written_as_single_items_is_read() -> None:
    metric = """_ma_qa_metric.id 1
_ma_qa_metric.name pLDDT
"""

    assert _scores(_document(metric, _LOCAL)) == {
        "1": 90.0,
        "2": 80.0,
    }


def test_a_single_declared_metric_other_than_plddt_is_not_read_as_one() -> None:
    metric = """_ma_qa_metric.id 2
_ma_qa_metric.name PAE
"""
    local = """loop_
_ma_qa_metric_local.label_seq_id
_ma_qa_metric_local.metric_id
_ma_qa_metric_local.metric_value
1 2 5.0
"""

    assert _scores(_document(metric, local)) == {}


def test_a_model_that_names_one_residue_twice_differently_is_rejected() -> None:
    renamed = _ALPHAFOLD_LAYOUT.replace("CA VAL 2 91.12", "CA GLY 2 91.12")

    assert parse_afdb_model_cif(_document(renamed)) == ({}, {})


def test_names_are_kept_only_for_scored_residues() -> None:
    atoms = """loop_
_atom_site.label_atom_id
_atom_site.label_comp_id
_atom_site.label_seq_id
_atom_site.B_iso_or_equiv
CA ALA 1 70.0
CA GLY 2 ?
"""

    assert parse_afdb_model_cif(_document(atoms)) == ({"1": 70.0}, {"1": "ALA"})
