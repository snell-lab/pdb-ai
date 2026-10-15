from __future__ import annotations

import dataclasses
import math
from typing import cast

import pandas as pd
import pytest

from pdbai.engines.sources.alphafold.annotation import (
    annotate_atom_dataframe,
    residue_names_match,
)
from pdbai.engines.sources.alphafold.models import (
    AFDB_STATUS_AVAILABLE,
    AFDB_STATUS_LOOKUP_FAILED,
    AFDB_STATUS_MODEL_NOT_FOUND,
    AFDB_STATUS_NO_UNIPROT_MAPPING,
    AFDBAlignedRegion,
    AFDBEntityReference,
    AFDBEntryResult,
    AFDBModelResult,
)


def _atoms(positions: list[str], *, entity: str = "1") -> pd.DataFrame:
    return pd.DataFrame(
        {
            "_atom_site.label_entity_id": [entity] * len(positions),
            "_atom_site.label_seq_id": positions,
            "res_name": ["ALA"] * len(positions),
            "_pdbai_chem_comp_type": [""] * len(positions),
        }
    )


def _entry(
    references: list[AFDBEntityReference],
    models: dict[str, AFDBModelResult],
) -> AFDBEntryResult:
    # The atoms of _atoms are alanines, so a model that names no residue is
    # given an alanine at every scored position, which lets its scores map.
    named_models = {
        accession: model
        if model.residue_name_by_position
        else dataclasses.replace(
            model,
            residue_name_by_position=dict.fromkeys(model.plddt_by_position, "ALA"),
        )
        for accession, model in models.items()
    }
    return AFDBEntryResult(
        pdb_id="1ABC",
        references=references,
        models=named_models,
        status=AFDB_STATUS_AVAILABLE,
        protein_entity_ids=["1"],
    )


def test_scores_broadcast_to_every_atom_of_a_residue() -> None:

    atoms = _atoms(["1", "1", "1", "2", "2"])
    entry = _entry(
        [
            AFDBEntityReference(
                entity_id="1",
                uniprot_id="P12345",
                aligned_regions=(AFDBAlignedRegion(1, 1, 2),),
            )
        ],
        {
            "P12345": AFDBModelResult(
                "P12345",
                AFDB_STATUS_AVAILABLE,
                plddt_by_position={"1": 91.5, "2": 44.25},
            )
        },
    )

    result = annotate_atom_dataframe(atoms, entry)

    assert result["afdb_plddt"].tolist() == [91.5, 91.5, 91.5, 44.25, 44.25]
    assert result["afdb_residue_has_plddt"].tolist() == [True] * 5
    assert result["uniprot_id"].tolist() == ["P12345"] * 5
    assert result["afdb_plddt_coverage"].tolist() == [1.0] * 5


def test_partial_coverage_counts_unique_residues() -> None:
    atoms = _atoms(["1", "1", "2", "3"])
    entry = _entry(
        [
            AFDBEntityReference(
                entity_id="1",
                uniprot_id="P12345",
                aligned_regions=(AFDBAlignedRegion(1, 1, 3),),
            )
        ],
        {
            "P12345": AFDBModelResult(
                "P12345", AFDB_STATUS_AVAILABLE, plddt_by_position={"2": 70.0}
            )
        },
    )

    result = annotate_atom_dataframe(atoms, entry)

    assert result["afdb_plddt"].isna().tolist() == [True, True, False, True]
    assert result["afdb_residue_has_plddt"].tolist() == [False, False, True, False]
    coverage = cast(list[float], result["afdb_plddt_coverage"].tolist())
    assert all(math.isclose(value, 1 / 3) for value in coverage)


def test_ambiguous_overlapping_references_stay_unassigned() -> None:

    atoms = _atoms(["1", "2"])
    references = [
        AFDBEntityReference(
            entity_id="1",
            uniprot_id=accession,
            aligned_regions=(AFDBAlignedRegion(1, 1, 1),),
        )
        for accession in ("P11111", "P22222")
    ]
    models = {
        accession: AFDBModelResult(
            accession, AFDB_STATUS_AVAILABLE, plddt_by_position={"1": 88.0}
        )
        for accession in ("P11111", "P22222")
    }

    result = annotate_atom_dataframe(atoms, _entry(references, models))

    assert result.loc[0, "uniprot_id"] is None
    assert pd.isna(result.loc[0, "afdb_plddt"])
    assert result["afdb_residue_has_plddt"].tolist() == [False, False]


def test_residue_provenance_uses_the_fragment_that_supplied_the_score() -> None:
    atoms = _atoms(["1", "2"])
    entry = _entry(
        [
            AFDBEntityReference(
                entity_id="1",
                uniprot_id="P12345",
                aligned_regions=(AFDBAlignedRegion(1, 1, 2),),
            )
        ],
        {
            "P12345": AFDBModelResult(
                "P12345",
                AFDB_STATUS_AVAILABLE,
                plddt_by_position={"1": 90.0, "2": 80.0},
                model_id_by_position={
                    "1": "AF-P12345-F1-model_v4",
                    "2": "AF-P12345-F2-model_v4",
                },
            )
        },
    )

    result = annotate_atom_dataframe(atoms, entry)

    assert result["afdb_model_id"].tolist() == [
        "AF-P12345-F1-model_v4",
        "AF-P12345-F2-model_v4",
    ]


def test_non_polymer_rows_are_left_out_of_the_mapping() -> None:
    atoms = _atoms(["1", "", "2"])
    entry = _entry(
        [
            AFDBEntityReference(
                entity_id="1",
                uniprot_id="P12345",
                aligned_regions=(AFDBAlignedRegion(1, 1, 2),),
            )
        ],
        {
            "P12345": AFDBModelResult(
                "P12345",
                AFDB_STATUS_AVAILABLE,
                plddt_by_position={"1": 90.0, "2": 80.0},
            )
        },
    )

    result = annotate_atom_dataframe(atoms, entry)

    assert pd.isna(result.loc[1, "afdb_plddt"])
    assert pd.isna(result.loc[1, "afdb_residue_has_plddt"])
    assert result["afdb_plddt"].tolist()[0] == 90.0


def test_model_without_scores_reports_status_without_claiming_a_mapping() -> None:
    atoms = _atoms(["1", "2"])
    entry = _entry(
        [
            AFDBEntityReference(
                entity_id="1",
                uniprot_id="P12345",
                aligned_regions=(AFDBAlignedRegion(1, 1, 2),),
            )
        ],
        {"P12345": AFDBModelResult("P12345", AFDB_STATUS_MODEL_NOT_FOUND)},
    )

    result = annotate_atom_dataframe(atoms, entry)

    assert result["afdb_status"].tolist() == [AFDB_STATUS_MODEL_NOT_FOUND] * 2
    assert result["uniprot_id"].tolist() == ["P12345"] * 2
    assert result["afdb_residue_has_plddt"].tolist() == [False, False]
    assert result["afdb_plddt_coverage"].tolist() == [0.0, 0.0]


def test_mapping_is_stable_regardless_of_atom_row_order() -> None:

    entry = _entry(
        [
            AFDBEntityReference(
                entity_id="1",
                uniprot_id="P12345",
                aligned_regions=(AFDBAlignedRegion(1, 1, 3),),
            )
        ],
        {
            "P12345": AFDBModelResult(
                "P12345",
                AFDB_STATUS_AVAILABLE,
                plddt_by_position={"1": 10.0, "2": 20.0, "3": 30.0},
            )
        },
    )

    ordered = annotate_atom_dataframe(_atoms(["1", "2", "3"]), entry)
    shuffled = annotate_atom_dataframe(_atoms(["3", "1", "2"]), entry)

    assert ordered["afdb_plddt"].tolist() == [10.0, 20.0, 30.0]
    assert shuffled["afdb_plddt"].tolist() == [30.0, 10.0, 20.0]


def test_capping_groups_do_not_count_against_plddt_coverage() -> None:
    # This case comes from 8S6V. Its entity 3 is nine amino acids followed by
    # an NH2 cap at label_seq_id 10. The fixture shortens it to three amino
    # acids and a cap at 4. The cap is a polymer position without a pLDDT,
    # and it is left out of the residue count, so a peptide whose every amino
    # acid is scored reports a coverage of 1.
    atoms = pd.DataFrame(
        {
            "_atom_site.label_entity_id": ["1"] * 4,
            "_atom_site.label_seq_id": ["1", "2", "3", "4"],
            "res_name": ["ALA", "PRO", "GLY", "NH2"],
            "_pdbai_chem_comp_type": [
                "L-peptide linking",
                "L-peptide linking",
                "peptide linking",
                "non-polymer",
            ],
        }
    )
    entry = _entry(
        [
            AFDBEntityReference(
                entity_id="1",
                uniprot_id="P12345",
                aligned_regions=(AFDBAlignedRegion(1, 1, 3),),
            )
        ],
        {
            "P12345": AFDBModelResult(
                "P12345",
                AFDB_STATUS_AVAILABLE,
                plddt_by_position={"1": 39.8, "2": 36.0, "3": 39.1},
                residue_name_by_position={"1": "ALA", "2": "PRO", "3": "GLY"},
            )
        },
    )

    result = annotate_atom_dataframe(atoms, entry)

    assert result["afdb_plddt_coverage"].tolist() == [1.0] * 4
    assert result["afdb_residue_has_plddt"].tolist()[:3] == [True, True, True]
    assert pd.isna(result["afdb_residue_has_plddt"].iloc[3])
    assert math.isnan(result["afdb_plddt"].iloc[3])
    assert result["afdb_status"].tolist() == [AFDB_STATUS_AVAILABLE] * 4


def test_unmapped_amino_acids_still_lower_coverage_when_types_are_absent() -> None:
    # Without chem_comp types the residue name decides, so a real amino acid
    # that received no score still counts as observed but unmapped.
    atoms = _atoms(["1", "2"])
    entry = _entry(
        [
            AFDBEntityReference(
                entity_id="1",
                uniprot_id="P12345",
                aligned_regions=(AFDBAlignedRegion(1, 1, 2),),
            )
        ],
        {
            "P12345": AFDBModelResult(
                "P12345",
                AFDB_STATUS_AVAILABLE,
                plddt_by_position={"1": 91.5},
            )
        },
    )

    result = annotate_atom_dataframe(atoms, entry)

    assert result["afdb_plddt_coverage"].tolist() == [0.5, 0.5]
    assert result["afdb_residue_has_plddt"].tolist() == [True, False]


def _aligned_entry(plddt: dict[str, float], names: dict[str, str]) -> AFDBEntryResult:
    # Entity positions 1 to 3 align to UniProt positions 1 to 3.
    return _entry(
        [
            AFDBEntityReference(
                entity_id="1",
                uniprot_id="P12345",
                aligned_regions=(AFDBAlignedRegion(1, 1, 3),),
            )
        ],
        {
            "P12345": AFDBModelResult(
                "P12345",
                AFDB_STATUS_AVAILABLE,
                plddt_by_position=plddt,
                residue_name_by_position=names,
            )
        },
    )


@pytest.mark.parametrize(
    ("deposited", "modelled", "expected"),
    [
        ("ALA", "ALA", True),
        (" ala ", "ALA", True),
        ("MSE", "MET", True),
        ("SEP", "SER", True),
        # Pyroglutamate forms from glutamine, though Gemmi's table says
        # glutamate. The other three are missing from that table.
        ("PCA", "GLN", True),
        ("PCA", "GLU", False),
        ("2MR", "ARG", True),
        ("DM0", "LYS", True),
        ("GLZ", "GLY", True),
        ("ALA", "GLY", False),
        ("MSE", "VAL", False),
        ("UNK", "ALA", False),
        ("", "ALA", False),
        ("ALA", "", False),
        ("NH2", "ALA", False),
    ],
)
def test_residue_names_match_by_standard_amino_acid(
    deposited: str, modelled: str, expected: bool
) -> None:
    assert residue_names_match(deposited, modelled) is expected


def test_a_residue_the_model_does_not_share_gets_no_score() -> None:
    # Position 2 is an engineered mutation, and position 3 is a residue the
    # model scores but does not name. Only position 1 is the model's residue.
    atoms = pd.DataFrame(
        {
            "_atom_site.label_entity_id": ["1"] * 3,
            "_atom_site.label_seq_id": ["1", "2", "3"],
            "res_name": ["ALA", "TRP", "GLY"],
            "_pdbai_chem_comp_type": ["", "", ""],
        }
    )
    entry = _aligned_entry({"1": 90.0, "2": 80.0, "3": 70.0}, {"1": "ALA", "2": "LEU"})

    result = annotate_atom_dataframe(atoms, entry)

    assert result["afdb_plddt"].tolist()[0] == 90.0
    assert result["afdb_plddt"].isna().tolist() == [False, True, True]
    assert result["afdb_residue_has_plddt"].tolist() == [True, False, False]
    assert result["afdb_model_id"].tolist()[1:] == [None, None]
    assert result["uniprot_id"].tolist() == ["P12345"] * 3
    coverage = cast(list[float], result["afdb_plddt_coverage"].tolist())
    assert all(math.isclose(value, 1 / 3) for value in coverage)


def test_a_modified_residue_takes_the_score_of_its_parent() -> None:
    atoms = pd.DataFrame(
        {
            "_atom_site.label_entity_id": ["1"],
            "_atom_site.label_seq_id": ["1"],
            "res_name": ["MSE"],
            "_pdbai_chem_comp_type": [""],
        }
    )

    result = annotate_atom_dataframe(atoms, _aligned_entry({"1": 90.0}, {"1": "MET"}))

    assert result["afdb_plddt"].tolist() == [90.0]
    assert result["afdb_residue_has_plddt"].tolist() == [True]


def test_a_position_recorded_with_two_amino_acids_needs_both_to_match() -> None:
    # Two chains of the entity disagree about position 1, as microheterogeneity
    # can leave them. The model can be the residue of only one of them.
    atoms = pd.DataFrame(
        {
            "_atom_site.label_entity_id": ["1", "1"],
            "_atom_site.label_seq_id": ["1", "1"],
            "res_name": ["ALA", "SER"],
            "_pdbai_chem_comp_type": ["", ""],
        }
    )

    result = annotate_atom_dataframe(atoms, _aligned_entry({"1": 90.0}, {"1": "ALA"}))

    assert result["afdb_plddt"].isna().all()
    assert result["afdb_residue_has_plddt"].tolist() == [False, False]


@pytest.mark.parametrize("label_seq_id", ["1.5", "0", "-1", "inf"])
def test_a_label_seq_id_that_is_not_a_position_maps_nothing(
    label_seq_id: str,
) -> None:
    # Truncating 1.5 would put position 1's score on this atom.
    atoms = pd.DataFrame(
        {
            "_atom_site.label_entity_id": ["1"],
            "_atom_site.label_seq_id": [label_seq_id],
            "res_name": ["ALA"],
            "_pdbai_chem_comp_type": [""],
        }
    )

    result = annotate_atom_dataframe(atoms, _aligned_entry({"1": 90.0}, {"1": "ALA"}))

    assert pd.isna(result.loc[0, "afdb_plddt"])
    assert pd.isna(result.loc[0, "afdb_residue_has_plddt"])
    assert pd.isna(result.loc[0, "afdb_status"])


def test_an_incomplete_discovery_keeps_an_answered_entity_unmapped() -> None:
    # RCSB answered that entity 2 is a protein without a UniProt mapping, and
    # could not be asked about entity 3.
    atoms = pd.DataFrame(
        {
            "_atom_site.label_entity_id": ["1", "2", "3"],
            "_atom_site.label_seq_id": ["1", "1", "1"],
            "res_name": ["ALA", "ALA", "ALA"],
            "_pdbai_chem_comp_type": ["", "", ""],
        }
    )
    entry = dataclasses.replace(
        _aligned_entry({"1": 90.0}, {"1": "ALA"}),
        discovery_complete=False,
        protein_entity_ids=["1", "2"],
        unmapped_protein_entity_ids=["2"],
    )

    result = annotate_atom_dataframe(atoms, entry)

    assert result["afdb_status"].tolist() == [
        AFDB_STATUS_AVAILABLE,
        AFDB_STATUS_NO_UNIPROT_MAPPING,
        AFDB_STATUS_LOOKUP_FAILED,
    ]
    assert result["afdb_residue_has_plddt"].tolist() == [True, False, False]


def test_every_atom_of_an_entity_carries_its_sequence_clusters() -> None:
    atoms = pd.concat(
        [
            _atoms(["1", "2"], entity="1"),
            _atoms(["1"], entity="2"),
            _atoms(["1"], entity="3"),
        ],
        ignore_index=True,
    )
    entry = dataclasses.replace(
        _entry([], {}),
        status=AFDB_STATUS_NO_UNIPROT_MAPPING,
        protein_entity_ids=["1", "2", "3"],
        sequence_clusters={"1": {"95": "391_95", "30": "700_30"}, "2": {"95": "8_95"}},
    )

    columns = annotate_atom_dataframe(atoms, entry)

    assert columns["sequence_cluster_95"].tolist() == ["391_95", "391_95", "8_95", None]
    assert columns["sequence_cluster_30"].tolist() == ["700_30", "700_30", None, None]
    assert columns["sequence_cluster_100"].isna().all()
    # A table without entity identifiers gets the columns, empty.
    bare = annotate_atom_dataframe(
        atoms.drop(columns=["_atom_site.label_entity_id"]), entry
    )
    assert bare["sequence_cluster_95"].isna().all()
